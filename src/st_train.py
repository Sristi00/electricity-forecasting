"""
Train and evaluate HeteroSTPriceForecaster (Heterogeneous Spatio-Temporal GNN).

Reuses the existing hetero_graph.pt; lag_to edges are excluded from the
forward pass — temporal dependencies are modelled by the in-model CausalTCN.

Evaluation: DK1 + DK2 test nodes only (same zones as XGBoost / hetero / homo).
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
from hetero_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR as ARTIFACTS_HETERO_DIR
from hetero_st_model import HeteroSTPriceForecaster


# ── Edge types used in spatial conv (lag_to handled by TCN) ──────────────────
SPATIAL_EDGE_TYPES = [
    ('hour',   'co_occurs_with', 'hour'),
    ('hour',   'belongs_to',     'market'),
    ('market', 'rev_belongs_to', 'hour'),
    ('market', 'interconnects',  'market'),
]


def st_train(
    hidden_channels: int  = 128,
    num_st_blocks:   int  = 2,
    max_epochs:      int  = 200,
    patience:        int  = 25,
    lr:              float = 0.001,
):
    print(f"\n⚡ ST-HeteroSAGE training | hidden={hidden_channels} | blocks={num_st_blocks}")
    ARTIFACTS_HETERO_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = GRAPH_DIR / "hetero_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found at {graph_path}. Run hetero_graph_builder.py first."
        )
    data = torch.load(graph_path, weights_only=False)
    data = data.to(DEVICE)
    print(f"   Graph loaded: {data['hour'].x.shape[0]:,} hour nodes | "
          f"{sum(data[e].edge_index.shape[1] for e in data.edge_types):,} total edges")

    # ── Filter to spatial-only edges (exclude lag_to) ─────────────────────────
    edge_index_dict = {et: data[et].edge_index for et in SPATIAL_EDGE_TYPES}
    spatial_edges   = sum(v.shape[1] for v in edge_index_dict.values())
    print(f"   Spatial edges (no lag_to): {spatial_edges:,}")

    # ── DK1+DK2-only masks ────────────────────────────────────────────────────
    T  = data['hour'].num_hours_per_zone   # 50399
    n2 = 2 * T                             # DK1 + DK2 node count
    dk12 = torch.zeros(4 * T, dtype=torch.bool, device=DEVICE)
    dk12[:n2] = True

    tr_mask   = data['hour'].train_mask.to(DEVICE) & dk12
    val_mask  = data['hour'].val_mask  .to(DEVICE) & dk12
    test_mask = data['hour'].test_mask .to(DEVICE) & dk12

    # Auxiliary mask: DE + HYDRO TRAINING nodes only (their data is real in the
    # 2020-2024 train window). Predicting them as a 0.1x side-task forces richer
    # cross-zone representations that feed DK1/DK2 via co_occurs/market edges.
    de_hydro_tr = data['hour'].train_mask.to(DEVICE) & (~dk12)

    # ── Target scaler ─────────────────────────────────────────────────────────
    scaler_path = GRAPH_DIR / "hetero_scalers.pkl"
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)['target_scaler']

    # Fit target scaler on DK1+DK2 raw targets from the training window
    y_all_raw = data['hour'].y.cpu().numpy()   # raw DKK (stored unscaled in hetero graph)
    y_train_dk12 = y_all_raw[tr_mask.cpu().numpy()]
    target_scaler.fit(y_train_dk12.reshape(-1, 1))

    y_scaled = torch.tensor(
        target_scaler.transform(y_all_raw.reshape(-1, 1)).flatten(),
        dtype=torch.float32, device=DEVICE
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = data['hour'].x.shape[1]
    model = HeteroSTPriceForecaster(
        in_channels     = in_channels,
        hidden_channels = hidden_channels,
        num_st_blocks   = num_st_blocks,
        temporal_dilations = (1, 4, 24),
        temporal_kernel    = 7,
    ).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {params:,} | in_channels={in_channels}")

    x_dict = {'hour': data['hour'].x, 'market': data['market'].x}

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    ckpt_path = ARTIFACTS_HETERO_DIR / "best_st_hetero_model.pt"
    best_val  = float('inf')
    pat_ctr   = 0
    t0        = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out       = model(x_dict, edge_index_dict)
        main_loss = F.mse_loss(out[tr_mask], y_scaled[tr_mask])
        aux_loss  = F.mse_loss(out[de_hydro_tr], y_scaled[de_hydro_tr])
        loss      = main_loss + 0.1 * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x_dict, edge_index_dict)
            val_loss = F.mse_loss(val_out[val_mask], y_scaled[val_mask]).item()
            # Report val MAE in DKK for readability
            val_pred_dkk = target_scaler.inverse_transform(
                val_out[val_mask].cpu().numpy().reshape(-1, 1)
            ).flatten()
            val_true_dkk = y_all_raw[val_mask.cpu().numpy()]
            val_mae_dkk  = float(mean_absolute_error(val_true_dkk, val_pred_dkk))

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():.4f} | "
                  f"Val MAE: {val_mae_dkk:.1f} DKK | {elapsed:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            pat_ctr = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("  Evaluating on test set...")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    with torch.no_grad():
        final_out = model(x_dict, edge_index_dict).cpu().numpy()

    tm_np = test_mask.cpu().numpy()
    y_pred_dkk = target_scaler.inverse_transform(
        final_out[tm_np].reshape(-1, 1)
    ).flatten()
    y_true_dkk = y_all_raw[tm_np]

    mae   = float(mean_absolute_error(y_true_dkk, y_pred_dkk))
    rmse  = float(np.sqrt(np.mean((y_pred_dkk - y_true_dkk) ** 2)))
    r2    = float(r2_score(y_true_dkk, y_pred_dkk))
    smape = float(np.mean(
        2 * np.abs(y_pred_dkk - y_true_dkk) /
        (np.abs(y_true_dkk) + np.abs(y_pred_dkk) + 1e-8)
    ) * 100)

    metrics = {
        "model":           "ST-HeteroSAGE",
        "hidden_channels": hidden_channels,
        "num_st_blocks":   num_st_blocks,
        "eval_zones":      "DK1+DK2",
        "mae":   round(mae,   4),
        "rmse":  round(rmse,  4),
        "r2":    round(r2,    4),
        "smape": round(smape, 4),
    }
    out_path = ARTIFACTS_HETERO_DIR / "st_hetero_metrics.json"
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📊 Test Results (DK1 + DK2):")
    print(f"   MAE:   {mae:.2f} DKK")
    print(f"   RMSE:  {rmse:.2f} DKK")
    print(f"   R²:    {r2:.4f}")
    print(f"   SMAPE: {smape:.2f}%")
    print(f"\n✅ Checkpoint → {ckpt_path}")
    print(f"   Metrics    → {out_path}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden",  type=int, default=128)
    ap.add_argument("--blocks",  type=int, default=2)
    ap.add_argument("--epochs",  type=int, default=200)
    ap.add_argument("--patience",type=int, default=25)
    args = ap.parse_args()
    st_train(
        hidden_channels = args.hidden,
        num_st_blocks   = args.blocks,
        max_epochs      = args.epochs,
        patience        = args.patience,
    )
