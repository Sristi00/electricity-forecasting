"""
Quick retrain script — trains HeteroPriceForecaster on the current graph
for up to 200 epochs with validation-based early stopping.

Targets are StandardScaler-normalized during training (same as homo GNN) and
inverse-transformed back to DKK for all reported metrics.
Overwrites best_hetero_model.pt when complete.
"""
import sys, json, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroPriceForecaster
from mlflow_config import setup_mlflow


def quick_retrain(hidden_channels=128, max_epochs=200, patience=25, warm_start=False):
    """
    Train HeteroPriceForecaster on the current graph.

    warm_start:
        False (default) — fresh random weights, full LR (0.002), full epoch
            budget. Use for a from-scratch retrain (e.g. after a full graph
            rebuild with a refit scaler).
        True — load existing best_hetero_model.pt weights and fine-tune with a
            gentler LR (5e-4) and a shorter epoch budget. Much faster / cheaper
            for incremental updates when new data arrives. Requires the saved
            checkpoint to match the current feature dimensionality and uses the
            checkpoint's hidden_channels automatically. Falls back to a fresh
            train if no checkpoint exists.
    """
    print("\n⚡ Quick retrain — HeteroPriceForecaster" + ("  (warm-start)" if warm_start else ""))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = ARTIFACTS_DIR / "best_hetero_model.pt"

    # ── Decide warm-start vs fresh + derive matching hyperparameters ───────────
    do_warm = warm_start and ckpt_path.exists()
    if warm_start and not ckpt_path.exists():
        print("   ⚠️  warm_start requested but no checkpoint found — training fresh.")

    if do_warm:
        # Read hidden_channels straight from the checkpoint so architecture matches.
        _ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hidden_channels = _ckpt['market_lin.weight'].shape[0]
        lr = 5e-4                              # gentle fine-tune LR
        max_epochs = min(max_epochs, 40)       # short budget for warm-start
        patience = min(patience, 10)
    else:
        _ckpt = None
        lr = 0.001  # match homo GNN lr for fair comparison

    # ── MLflow setup ───────────────────────────────────────────────────────────
    try:
        setup_mlflow()
        import mlflow
        _mlflow_run = mlflow.start_run(run_name="HeteroSAGE-warmstart" if do_warm else "HeteroSAGE")
        _mlflow_ok = True
    except Exception as _e:
        print(f"   [MLflow] Disabled — {_e}")
        _mlflow_ok = False

    try:
        # ── Log hyperparameters ────────────────────────────────────────────────
        if _mlflow_ok:
            try:
                mlflow.log_params({
                    "hidden_channels": hidden_channels,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": 1e-4,
                    "warm_start": do_warm,
                })
            except Exception:
                pass

        torch.manual_seed(42)
        np.random.seed(42)

        data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
        num_hours = int(data['hour'].num_hours_per_zone)

        model = HeteroPriceForecaster(
            metadata=data.metadata(),
            hour_in_features=data['hour'].x.shape[1],
            hidden_channels=hidden_channels,
        ).to(DEVICE)

        # ── Warm-start: load existing weights before training ──────────────────
        if do_warm:
            try:
                model.load_state_dict(_ckpt)
                print(f"   🔁 Warm-start — loaded existing weights, fine-tuning at lr={lr}")
            except Exception as _e:
                print(f"   ⚠️  Could not load checkpoint for warm-start ({_e}) — training fresh.")
                do_warm = False
                lr = 0.002

        params = sum(p.numel() for p in model.parameters())
        print(f"   Params: {params:,} | hidden_channels={hidden_channels} | {num_hours} hours/zone")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        x_dict  = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        ei      = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
        y_np    = data['hour'].y.cpu().numpy()
        tr_mask_full = data['hour'].train_mask.to(DEVICE)
        vl_mask_full = data['hour'].val_mask.to(DEVICE)
        te_mask_full = data['hour'].test_mask.to(DEVICE)

        # Restrict loss, val, and test evaluation to DK1+DK2 only — matching the
        # homo GNN which also trains/evaluates on DK1+DK2 exclusively.
        # HYDRO and DE have all-zero prices (placeholder data); including them in
        # the loss contaminates gradients and halves the effective training signal.
        # All 4 zones still participate in message-passing for graph structure.
        n2 = 2 * num_hours  # first 2×N nodes = DK1 + DK2
        dk12 = torch.zeros(4 * num_hours, dtype=torch.bool, device=DEVICE)
        dk12[:n2] = True
        tr_mask = tr_mask_full & dk12
        vl_mask = vl_mask_full & dk12
        te_mask = te_mask_full & dk12

        # Scale y targets: fit on DK1+DK2 training nodes only (real prices).
        # Using all-zone y would contaminate the scaler with ~50% zeros.
        y_scaler = StandardScaler()
        dk12_np = dk12.cpu().numpy()
        tr_np   = data['hour'].train_mask.cpu().numpy()
        y_scaler.fit(y_np[tr_np & dk12_np].reshape(-1, 1))
        y_scaled = y_scaler.transform(y_np.reshape(-1, 1)).ravel()
        y = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)

        best_val  = float('inf')
        patience_ctr = 0
        t0 = time.time()

        # DE + HYDRO training mask for the auxiliary reconstruction loss.
        # These zones now have real prices; predicting them forces the model to
        # build representations that capture cross-zone price dynamics, which
        # then propagate back to DK1/DK2 via the co_occurs_with and market edges.
        de_hydro = ~dk12
        de_hydro_tr = tr_mask_full & de_hydro

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out       = model(x_dict, ei, num_hours=num_hours).view(-1)
            main_loss = F.mse_loss(out[tr_mask], y[tr_mask])
            aux_loss  = F.mse_loss(out[de_hydro_tr], y[de_hydro_tr])
            loss      = main_loss + 0.1 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out  = model(x_dict, ei, num_hours=num_hours).view(-1)
                val_loss = F.mse_loss(val_out[vl_mask], y[vl_mask]).item()
                # Inverse-transform to DKK for readable reporting (DK1+DK2 only)
                vl_np  = vl_mask.cpu().numpy()
                vp_dkk = y_scaler.inverse_transform(val_out[vl_mask].cpu().numpy().reshape(-1, 1)).ravel()
                ya_dkk = y_np[vl_np]
                val_mae = float(mean_absolute_error(ya_dkk, vp_dkk))

            scheduler.step(val_loss)

            # Log per-epoch metrics to MLflow
            if _mlflow_ok:
                try:
                    mlflow.log_metric("train_mse", loss.item(), step=epoch)
                    mlflow.log_metric("val_mae", val_mae, step=epoch)
                except Exception:
                    pass

            if epoch == 1 or epoch % 10 == 0:
                elapsed = time.time() - t0
                print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():10.4f} | Val MAE: {val_mae:6.1f} DKK | {elapsed:.0f}s")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break

        # ── Final test evaluation ──────────────────────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        model.eval()
        with torch.no_grad():
            final_scaled = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()

        tm_np = te_mask.cpu().numpy()
        y_te  = y_np[tm_np]
        # Inverse-transform scaled predictions → DKK
        p_te  = y_scaler.inverse_transform(final_scaled[tm_np].reshape(-1, 1)).ravel()

        smape = float(np.mean(2 * np.abs(p_te - y_te) / (np.abs(y_te) + np.abs(p_te) + 1e-8)) * 100)
        metrics = {
            "eval_zones": "DK1+DK2",
            "mae":   float(mean_absolute_error(y_te, p_te)),
            "rmse":  float(np.sqrt(np.mean((p_te - y_te)**2))),
            "r2":    float(r2_score(y_te, p_te)),
            "smape": smape,
        }
        print(f"\n📊 Test Results (DK1 + DK2 — same evaluation zones as homo GNN):")
        print(f"   MAE:   {metrics['mae']:.2f} DKK")
        print(f"   RMSE:  {metrics['rmse']:.2f} DKK")
        print(f"   R²:    {metrics['r2']:.4f}")
        print(f"   SMAPE: {metrics['smape']:.2f}%")

        # Log final test metrics and checkpoint artifact
        if _mlflow_ok:
            try:
                mlflow.log_metrics({
                    "test_mae":   metrics["mae"],
                    "test_rmse":  metrics["rmse"],
                    "test_r2":    metrics["r2"],
                    "test_smape": metrics["smape"],
                })
                mlflow.log_artifact(str(ckpt_path))
            except Exception:
                pass

        with open(ARTIFACTS_DIR / "hetero_metrics_clean.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Persist scaler parameters so analysis scripts can inverse-transform.
        scaler_params = {"mean": float(y_scaler.mean_[0]), "scale": float(y_scaler.scale_[0])}
        with open(ARTIFACTS_DIR / "y_scaler.json", "w") as f:
            json.dump(scaler_params, f)

        print(f"\n✅ Checkpoint saved → {ckpt_path}")
        return metrics

    finally:
        if _mlflow_ok:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--hidden", type=int, default=128)
    args, _ = ap.parse_known_args()
    quick_retrain(hidden_channels=args.hidden, warm_start=args.warm_start)
