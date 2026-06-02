"""
GNN Configuration - All settings for Graph Neural Network based electricity forecasting
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
GRAPH_DIR = DATA_DIR / "graphs"
DB_PATH = DATA_DIR / "energy.db"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

# ── API Settings ───────────────────────────────────────────────────
# Energy-Charts API - free, no key required
ENERGY_CHARTS_API = "https://api.energy-charts.info/price"

# Open-Meteo - free weather data
OPENMETEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Denmark coordinates (Aarhus area)
DENMARK_LAT = 56.16
DENMARK_LON = 10.20

# Price zones
PRICE_ZONES = ["DK1", "DK2"]
DEFAULT_ZONE = "DK1"

# EUR to DKK conversion
EUR_TO_DKK = 7.46

# Historical data to fetch (days)
HISTORICAL_DAYS = 90

# ── Graph Construction Settings ────────────────────────────────────
# Edge types - day-ahead-safe temporal connections.
# Only lags >= 24h are used: these are known at gate closure (noon D-1) and
# therefore valid for genuine day-ahead forecasting. The previous t-1/t-2/t-6
# edges leaked near-term prices and were removed.
EDGE_TYPES = {
    "temporal_24h": 24,    # Connect same hour previous day
    "temporal_168h": 168,  # Connect same hour previous week
}

# Node feature configuration
NODE_FEATURES = [
    # Time features
    "hour", "day_of_week", "month", "is_weekend",
    # Lag features
    "price_lag_1h", "price_lag_2h", "price_lag_24h",
    "price_lag_48h", "price_lag_168h",
    # Rolling statistics
    "price_rolling_6h_mean", "price_rolling_12h_mean",
    "price_rolling_24h_mean", "price_rolling_6h_std",
    "price_rolling_24h_std",
    # Weather features
    "temperature_c", "wind_speed_ms", "cloud_cover_pct", "humidity_pct",
]

# ── GNN Model Settings ─────────────────────────────────────────────
GNN_CONFIG = {
    # Model architecture
    "hidden_channels": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "conv_type": "GCN",  # Options: GCN, GAT, GraphSAGE, GIN
    
    # GAT-specific
    "num_heads": 4,
    
    # Training
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "batch_size": 32,
    "num_epochs": 100,
    "early_stopping_patience": 15,
    
    # Data split
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# ── Prediction Settings ────────────────────────────────────────────
FORECAST_HORIZON = 24  # Predict next 24 hours

# ── Device Settings ────────────────────────────────────────────────
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 Using device: {DEVICE}")