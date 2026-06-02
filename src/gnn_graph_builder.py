"""
Homogeneous spatial-temporal graph builder for multi-area electricity price forecasting.

Node layout: interleaved — node index = t*4 + zone_offset
  zone offsets: 0=DK1, 1=DK2, 2=DE, 3=HYDRO

All heavy loops are vectorised with numpy; no Python for-loop over timesteps.
Train/val/test masks target DK1 + DK2 only (DE/HYDRO are zero-filled in DB).
13 features per node to match hetero model feature richness.
"""
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from gnn_config import EDGE_TYPES, GRAPH_DIR
from hetero_pipeline import prepare_multi_area_data


class TemporalGraphBuilder:
    """Builds a flat homogeneous spatial-temporal mesh for fair benchmarking."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def build_graph(self, train_mask_ratio: float = 0.8, val_mask_ratio: float = 0.1) -> Data:
        print("\n Building Multi-Zone Spatial-Temporal Mesh Graph (vectorised)...")

        df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()
        num_hours = len(df_dk1)
        num_nodes = num_hours * 4  # 4 zones per timestep

        # ── Feature columns (from data) ───────────────────────────────────────
        lag_weather_cols = [
            'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
            'price_rolling_24h_mean', 'price_rolling_24h_std',
            'temperature_c', 'wind_speed_ms', 'cloud_cover_pct', 'humidity_pct',
            'load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk',
        ]
        # Cyclical time features (same for all zones) — 4 extra columns
        ts = pd.to_datetime(df_dk1['timestamp'])
        hour_of_day = ts.dt.hour.values.astype(np.float32)
        day_of_week = ts.dt.dayofweek.values.astype(np.float32)
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos  = np.cos(2 * np.pi * day_of_week / 7)
        # shape: (num_hours, 4)
        cyclic_block = np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=1)

        num_features = len(lag_weather_cols) + 4  # 17 total (13 lag/weather/fund + 4 cyclical)

        # ── Vectorised feature + target extraction ────────────────────────────
        # zone_offset: 0=DK1, 1=DK2, 2=DE, 3=HYDRO (interleaved layout)
        raw_x = np.zeros((num_nodes, num_features), dtype=np.float32)
        raw_y = np.zeros(num_nodes, dtype=np.float32)

        zone_dfs = [df_dk1, df_dk2, df_de, df_hydro]
        for z_off, df in enumerate(zone_dfs):
            node_idx = np.arange(num_hours) * 4 + z_off  # shape (num_hours,)
            for col_i, col in enumerate(lag_weather_cols):
                if col in df.columns:
                    raw_x[node_idx, col_i] = df[col].fillna(0.0).values.astype(np.float32)
            # Cyclical features appended after lag/weather columns
            raw_x[node_idx, len(lag_weather_cols):] = cyclic_block
            if 'price_dkk' in df.columns:
                raw_y[node_idx] = df['price_dkk'].fillna(0.0).values.astype(np.float32)

        # ── Scalers ───────────────────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(raw_x)

        train_hours = int(num_hours * train_mask_ratio)
        t_train = np.arange(train_hours)
        # Fit target scaler strictly on DK1+DK2 training nodes (exclude zero-filled zones)
        dk12_train_idx = np.concatenate([t_train * 4 + 0, t_train * 4 + 1])
        self.target_scaler.fit(raw_y[dk12_train_idx].reshape(-1, 1))
        Y_scaled = self.target_scaler.transform(raw_y.reshape(-1, 1)).flatten()

        # ── Vectorised edge construction ───────────────────────────────────────
        t_range = np.arange(num_hours)
        base_t  = t_range * 4
        all_src, all_tgt = [], []

        # Spatial: at every timestep, connect zone pairs bidirectionally
        spatial_pairs = [(0, 1), (1, 0), (0, 2), (2, 0),
                         (1, 2), (2, 1), (1, 3), (3, 1)]
        for src_off, tgt_off in spatial_pairs:
            all_src.append(base_t + src_off)
            all_tgt.append(base_t + tgt_off)

        # Temporal: connect t→t+lag and t+lag→t for each zone
        for _etype, lag in EDGE_TYPES.items():
            t_valid   = t_range[t_range >= lag]
            old_base  = (t_valid - lag) * 4
            new_base  = t_valid * 4
            for z in range(4):
                all_src.append(old_base + z)
                all_tgt.append(new_base + z)
                all_src.append(new_base + z)
                all_tgt.append(old_base + z)

        all_src_np = np.concatenate(all_src)
        all_tgt_np = np.concatenate(all_tgt)
        edge_index = torch.tensor(np.stack([all_src_np, all_tgt_np]), dtype=torch.long)

        # ── Masks (DK1 + DK2 only for train/val/test) ─────────────────────────
        val_hours  = int(num_hours * val_mask_ratio)
        t_val      = np.arange(train_hours, train_hours + val_hours)
        t_test     = np.arange(train_hours + val_hours, num_hours)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

        for z_off in [0, 1]:  # DK1=0, DK2=1
            train_mask[t_train * 4 + z_off] = True
            val_mask  [t_val   * 4 + z_off] = True
            test_mask [t_test  * 4 + z_off] = True

        data = Data(
            x=torch.FloatTensor(X_scaled),
            edge_index=edge_index,
            y=torch.FloatTensor(Y_scaled),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        data.num_nodes  = num_nodes
        data.num_hours  = num_hours
        print(f"   Graph: {num_nodes:,} nodes | {edge_index.shape[1]:,} edges | {num_features} features")
        return data

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'feature_scaler': self.scaler, 'target_scaler': self.target_scaler}, f)


def load_graph(graph_path: Path = None) -> Data:
    if graph_path is None:
        graph_path = GRAPH_DIR / "temporal_graph.pt"
    return torch.load(graph_path, weights_only=False)


def build_and_save_graph() -> Data:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    builder = TemporalGraphBuilder()
    graph_data = builder.build_graph()
    torch.save(graph_data, GRAPH_DIR / "temporal_graph.pt")
    builder.save(GRAPH_DIR / "scaler.pkl")
    print("Graph saved to", GRAPH_DIR / "temporal_graph.pt")
    return graph_data


if __name__ == "__main__":
    build_and_save_graph()
