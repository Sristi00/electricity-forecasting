import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts_hetero"
GRAPH_DIR = DATA_DIR / "graphs_hetero"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# Heterogeneous Graph Settings
NODE_TYPES = ['zone_hour', 'weather']
EDGE_TYPES = [
    ('zone_hour', 'temporal', 'zone_hour'),
    ('zone_hour', 'spatial', 'zone_hour'),
    ('zone_hour', 'affected_by', 'weather'),
    ('weather', 'influences', 'zone_hour')
]

HETERO_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "weight_decay": 5e-4,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Heterogeneous Config Loaded | Device: {DEVICE}")


def load_y_scaler():
    """
    Return (mean, scale) for the training-split StandardScaler saved by quick_retrain.
    Returns (0.0, 1.0) as identity if the file doesn't exist (pre-scaled checkpoints).
    """
    import json
    p = ARTIFACTS_DIR / "y_scaler.json"
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        return float(s["mean"]), float(s["scale"])
    return 0.0, 1.0


def inverse_scale_y(arr, mean, scale):
    """Inverse-transform scaled predictions back to DKK."""
    import numpy as np
    return np.asarray(arr) * scale + mean