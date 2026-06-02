"""
Rebuild homogeneous graph on the full dataset and retrain GraphSAGE.

Uses the same 80/10/10 chronological split as the hetero model.
Evaluates on DK1 + DK2 only (same zones as XGBoost and hetero model).
Saves real metrics to artifacts/homo_gnn_metrics.json for compare_models.py.
"""
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from gnn_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR, GNN_CONFIG
from gnn_graph_builder import build_and_save_graph, load_graph
from gnn_models import GraphSAGEModel
from mlflow_config import setup_mlflow


def homo_retrain(hidden_channels=128, max_epochs=200, patience=25, rebuild_graph=True):
    print("\n Homogeneous GNN Retrain (GraphSAGE, full 5.7-year dataset)")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ───────────────────────────────────────────────────────────
    try:
        setup_mlflow()
        import mlflow
        _mlflow_run = mlflow.start_run(run_name="HomoGNN-GraphSAGE")
        _mlflow_ok = True
    except Exception as _e:
        print(f"   [MLflow] Disabled — {_e}")
        _mlflow_ok = False

    try:
        # Log hyperparameters
        if _mlflow_ok:
            try:
                mlflow.log_params({
                    "hidden_channels": hidden_channels,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "lr": 0.001,
                    "weight_decay": 5e-4,
                })
            except Exception:
                pass

        # ── Build or load graph ────────────────────────────────────────────────────
        if rebuild_graph or not (GRAPH_DIR / "temporal_graph.pt").exists():
            print("  [1/3] Rebuilding homogeneous graph (vectorised)...")
            data = build_and_save_graph()
        else:
            print("  [1/3] Loading existing graph...")
            data = load_graph()

        num_hours  = data.num_hours
        num_nodes  = data.num_nodes
        num_feats  = data.x.shape[1]
        n_train    = int(data.train_mask.sum())
        n_test     = int(data.test_mask.sum())
        print(f"   Hours: {num_hours:,} | Nodes: {num_nodes:,} | Features: {num_feats}")
        print(f"   Train nodes (DK1+DK2): {n_train:,} | Test nodes: {n_test:,}")

        # ── Model ──────────────────────────────────────────────────────────────────
        print("  [2/3] Training GraphSAGE...")
        model = GraphSAGEModel(
            num_features=num_feats,
            hidden_channels=hidden_channels,
            num_layers=3,
            dropout=0.0,  # BN + dropout conflict: BN already regularises
        ).to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"   Params: {params:,} | hidden_channels={hidden_channels}")

        data_dev = data.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        ckpt_path = ARTIFACTS_DIR / "best_homo_model.pt"
        best_val  = float('inf')
        pat_ctr   = 0
        t0        = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out  = model(data_dev.x, data_dev.edge_index)
            loss = F.mse_loss(out[data_dev.train_mask], data_dev.y[data_dev.train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out  = model(data_dev.x, data_dev.edge_index)
                val_loss = F.mse_loss(val_out[data_dev.val_mask], data_dev.y[data_dev.val_mask]).item()
                val_mae  = F.l1_loss(val_out[data_dev.val_mask], data_dev.y[data_dev.val_mask]).item()

            scheduler.step(val_loss)

            # Log per-epoch metrics to MLflow
            if _mlflow_ok:
                try:
                    mlflow.log_metric("train_mse", loss.item(), step=epoch)
                    mlflow.log_metric("val_mae", val_mae, step=epoch)
                except Exception:
                    pass

            if epoch == 1 or epoch % 20 == 0:
                elapsed = time.time() - t0
                print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():10.2f} | Val MAE: {val_mae:.4f} | {elapsed:.0f}s")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
                pat_ctr = 0
            else:
                pat_ctr += 1
                if pat_ctr >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break

        # ── Test evaluation ────────────────────────────────────────────────────────
        print("  [3/3] Evaluating on test set...")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        model.eval()

        with torch.no_grad():
            final_scaled = model(data_dev.x, data_dev.edge_index).cpu().numpy()

        # Inverse-transform back to DKK
        scaler_pkl = GRAPH_DIR / "scaler.pkl"
        with open(scaler_pkl, 'rb') as f:
            scalers = pickle.load(f)
        target_scaler = scalers['target_scaler']

        tm_np = data.test_mask.cpu().numpy()
        y_scaled  = data.y.cpu().numpy()

        y_pred_dkk = target_scaler.inverse_transform(
            final_scaled[tm_np].reshape(-1, 1)
        ).flatten()
        y_true_dkk = target_scaler.inverse_transform(
            y_scaled[tm_np].reshape(-1, 1)
        ).flatten()

        mae   = float(mean_absolute_error(y_true_dkk, y_pred_dkk))
        rmse  = float(np.sqrt(np.mean((y_pred_dkk - y_true_dkk) ** 2)))
        r2    = float(r2_score(y_true_dkk, y_pred_dkk))
        smape = float(np.mean(
            2 * np.abs(y_pred_dkk - y_true_dkk) / (np.abs(y_true_dkk) + np.abs(y_pred_dkk) + 1e-8)
        ) * 100)
        mape  = float(np.mean(
            np.abs((y_pred_dkk - y_true_dkk) / (np.abs(y_true_dkk) + 1e-8))
        ) * 100)

        metrics = {
            "model":          "GraphSAGE (homogeneous)",
            "dataset_hours":  int(num_hours),
            "train_split":    "80%",
            "val_split":      "10%",
            "test_split":     "10%",
            "eval_zones":     "DK1 + DK2",
            "hidden_channels": hidden_channels,
            "mae":   round(mae,   4),
            "rmse":  round(rmse,  4),
            "r2":    round(r2,    4),
            "smape": round(smape, 4),
            "mape":  round(mape,  4),
        }

        # Log final test metrics and checkpoint artifact
        if _mlflow_ok:
            try:
                mlflow.log_metrics({
                    "test_mae":   mae,
                    "test_rmse":  rmse,
                    "test_r2":    r2,
                    "test_smape": smape,
                })
                mlflow.log_artifact(str(ckpt_path))
            except Exception:
                pass

        out_path = ARTIFACTS_DIR / "homo_gnn_metrics.json"
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n Test Results (DK1 + DK2 — same zones as XGBoost / Hetero):")
        print(f"   MAE:   {mae:.2f} DKK")
        print(f"   RMSE:  {rmse:.2f} DKK")
        print(f"   R²:    {r2:.4f}")
        print(f"   SMAPE: {smape:.2f}%")
        print(f"\n Checkpoint: {ckpt_path}")
        print(f" Metrics:    {out_path}")
        return metrics

    finally:
        if _mlflow_ok:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    homo_retrain(rebuild_graph=True)
