# OVERWRITE EXACTLY: src/hetero_train.py
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroPriceForecaster

def train_hetero_pipeline():
    print("\n🚀 Starting Training Loop for Upgraded Multi-Area Heterogeneous GNN...")
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)

    model = HeteroPriceForecaster(metadata=data.metadata(), hour_in_features=data['hour'].x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    y_raw = data['hour'].y.to(DEVICE)
    train_mask, val_mask, test_mask = data['hour'].train_mask, data['hour'].val_mask, data['hour'].test_mask
    HETERO_CHECKPOINT = ARTIFACTS_DIR / "best_hetero_model.pt"

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, num_hours=num_hours).view(-1)
        loss = F.mse_loss(out[train_mask], y_raw[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train MSE: {loss.item():8.2f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        final_out = model(data.x_dict, data.edge_index_dict, num_hours=num_hours).view(-1).cpu().numpy()
    
    metrics = {
        "mae": float(mean_absolute_error(y_raw[test_mask].cpu(), final_out[test_mask.cpu()])),
        "r2": float(r2_score(y_raw[test_mask].cpu(), final_out[test_mask.cpu()]))
    }
    
    with open(ARTIFACTS_DIR / "hetero_metrics_clean.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n✅ Training Complete. Metrics saved.")

if __name__ == "__main__":
    train_hetero_pipeline()