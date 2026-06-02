# OVERWRITE EXACTLY: src/gat_train.py
"""
[FOOTNOTE EXPERIMENT — NOT A PRIMARY MODEL]

Training loop for the Heterogeneous Graph Attention Network (HeteroGAT).

This is retained only as a documented negative result. Under a fair
evaluation (StandardScaler targets, DK1+DK2-only test set — identical to
HeteroSAGE and the homogeneous GNN), HeteroGAT scored **179 DKK MAE vs
HeteroSAGE's 167 DKK** — strictly worse. The ablation study independently
shows the cross-border `interconnects` edges (precisely what attention would
re-weight) contribute only ~+1 DKK, so there is almost nothing for an
attention mechanism to learn over plain GraphSAGE mean/sum aggregation.

Conclusion for the thesis: attention does not improve over GraphSAGE
aggregation for this graph; the primary heterogeneous model is HeteroSAGE.
HeteroGAT is excluded from the headline comparison (compare_models.py) and
kept here purely for reproducibility of that finding.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroGATPriceForecaster


def train_gat_pipeline():
    """
    Train HeteroGATPriceForecaster on DK1+DK2 only — same evaluation
    protocol as HomoGNN and HeteroPriceForecaster for a fair comparison.

    Targets are StandardScaler-normalised during training (prevents MSE
    loss from operating on raw DKK^2 values ≈ 250,000, which destabilises
    gradients) and inverse-transformed for all reported metrics.
    """
    print("\n⚡ Initializing Multi-Head HeteroGAT Training Pipeline...")

    torch.manual_seed(42)
    np.random.seed(42)

    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)

    model = HeteroGATPriceForecaster(
        metadata=data.metadata(),
        hour_in_features=data['hour'].x.shape[1],
        hidden_channels=128,
        heads=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture: HeteroGAT (4 heads) | Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    y_np = data['hour'].y.cpu().numpy()

    # Restrict to DK1+DK2 only — same as HomoGNN and HeteroPriceForecaster.
    n2 = 2 * num_hours
    dk12 = torch.zeros(4 * num_hours, dtype=torch.bool, device=DEVICE)
    dk12[:n2] = True

    tr_mask = data['hour'].train_mask.to(DEVICE) & dk12
    vl_mask = data['hour'].val_mask.to(DEVICE)   & dk12
    te_mask = data['hour'].test_mask.to(DEVICE)  & dk12

    # Scale y on DK1+DK2 training nodes only.
    dk12_np = dk12.cpu().numpy()
    tr_np   = data['hour'].train_mask.cpu().numpy()
    y_scaler = StandardScaler()
    y_scaler.fit(y_np[tr_np & dk12_np].reshape(-1, 1))
    y_scaled = y_scaler.transform(y_np.reshape(-1, 1)).ravel()
    y = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)

    x_dict          = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    best_val_loss = float('inf')
    patience_counter = 0
    GAT_CHECKPOINT = ARTIFACTS_DIR / "best_gat_model.pt"

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out  = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1)
        loss = F.mse_loss(out[tr_mask], y[tr_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1)
            val_loss = F.mse_loss(val_out[vl_mask], y[vl_mask]).item()
            vl_np    = vl_mask.cpu().numpy()
            vp_dkk   = y_scaler.inverse_transform(val_out[vl_mask].cpu().numpy().reshape(-1, 1)).ravel()
            val_mae  = float(mean_absolute_error(y_np[vl_np], vp_dkk))

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():10.4f} | Val MAE: {val_mae:6.1f} DKK")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), GAT_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"   Early stopping at epoch {epoch}.")
                break

    print("\n📊 Evaluating on test set (best checkpoint, DK1+DK2 only)...")
    model.load_state_dict(torch.load(GAT_CHECKPOINT, map_location=DEVICE, weights_only=False))
    model.eval()

    with torch.no_grad():
        final_scaled = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1).cpu().numpy()

    tm_np   = te_mask.cpu().numpy()
    y_test  = y_np[tm_np]
    p_test  = y_scaler.inverse_transform(final_scaled[tm_np].reshape(-1, 1)).ravel()

    smape = float(np.mean(
        2.0 * np.abs(p_test - y_test) / (np.abs(y_test) + np.abs(p_test) + 1e-8)
    ) * 100)

    metrics = {
        "eval_zones": "DK1+DK2",
        "mae":   float(mean_absolute_error(y_test, p_test)),
        "rmse":  float(np.sqrt(np.mean((p_test - y_test) ** 2))),
        "r2":    float(r2_score(y_test, p_test)),
        "smape": smape,
    }

    print(f"   MAE   : {metrics['mae']:.2f} DKK")
    print(f"   RMSE  : {metrics['rmse']:.2f} DKK")
    print(f"   R²    : {metrics['r2']:.4f}")
    print(f"   SMAPE : {metrics['smape']:.2f}%")

    with open(ARTIFACTS_DIR / "gat_metrics_clean.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ GAT training complete. Metrics saved to {ARTIFACTS_DIR}/gat_metrics_clean.json")


if __name__ == "__main__":
    train_gat_pipeline()
