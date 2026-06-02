# OVERWRITE EXACTLY: src/hetero_graph_builder.py
"""
Advanced Graph Builder for Heterogeneous Multi-Area Electricity Forecasting.
Implements Cyclical Calendar Profiles and Weighted Spatial Interconnect Features.
"""
import sys
import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

# Clean, working configuration imports
from hetero_config import GRAPH_DIR
from hetero_pipeline import prepare_multi_area_data 

def build_heterogeneous_spatiotemporal_graph(freeze_scaler=False):
    """
    Build the heterogeneous market graph.

    freeze_scaler:
        False (default) — fit fresh StandardScalers on the current training
            partition. Use for a full retrain from scratch.
        True — reuse the previously saved feature/target scalers from
            hetero_scalers.pkl instead of refitting. This keeps the input
            feature distribution fixed across incremental graph rebuilds so
            that warm-started model weights remain valid. Falls back to
            fitting fresh scalers if no saved pickle exists.
    """
    print("\n🏗️ Building Context-Aware Heterogeneous Market Graph...")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Leakage-Proof Clean Data Frames (Bypasses manual DB mapping)
    df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()
    
    # Sort to enforce chronological timeline validation integrity
    df_dk1 = df_dk1.sort_values('timestamp').reset_index(drop=True)
    df_dk2 = df_dk2.sort_values('timestamp').reset_index(drop=True)
    df_de = df_de.sort_values('timestamp').reset_index(drop=True)
    df_hydro = df_hydro.sort_values('timestamp').reset_index(drop=True)
    
    num_hours = len(df_dk1)
    
    # 2. Extract and Align Target Labels Matrix
    y_dk1 = df_dk1['price_dkk'].values.astype(np.float32)
    y_dk2 = df_dk2['price_dkk'].values.astype(np.float32)
    y_de = df_de['price_dkk'].values.astype(np.float32)
    y_hydro = df_hydro['price_dkk'].values.astype(np.float32)
    
    # 3. Feature Engineering: Cyclical Calendar Node Profiles
    timestamps = pd.to_datetime(df_dk1['timestamp'])
    hour_of_day = timestamps.dt.hour.values
    day_of_week = timestamps.dt.dayofweek.values
    
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
    week_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    week_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    
    def extract_features(df):
        cols = [
            'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
            'price_rolling_24h_mean', 'price_rolling_24h_std',
            'temperature_c', 'wind_speed_ms', 'cloud_cover_pct', 'humidity_pct',
            'load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk'
        ]
        # Return 0 defaults if weather variables aren't present in specific zone views
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[cols].values.astype(np.float32)

    x_dk1_base = extract_features(df_dk1)
    x_dk2_base = extract_features(df_dk2)
    x_de_base = extract_features(df_de)
    x_hydro_base = extract_features(df_hydro)
    
    # Append cyclical components directly to the feature arrays
    cyclical_stack = np.stack([hour_sin, hour_cos, week_sin, week_cos], axis=1).astype(np.float32)
    x_dk1 = np.hstack([x_dk1_base, cyclical_stack])
    x_dk2 = np.hstack([x_dk2_base, cyclical_stack])
    x_de = np.hstack([x_de_base, cyclical_stack])
    x_hydro = np.hstack([x_hydro_base, cyclical_stack])

    # 4. Fit Scalers sequentially to avoid cross-boundary distribution contamination
    train_idx_limit = int(num_hours * 0.8)

    scalers_path = GRAPH_DIR / "hetero_scalers.pkl"
    if freeze_scaler and scalers_path.exists():
        # Reuse previously fitted scalers so the feature distribution stays
        # fixed — required for stable warm-start fine-tuning on new data.
        with open(scalers_path, "rb") as f:
            _saved = pickle.load(f)
        feature_scaler = _saved['feature_scaler']
        target_scaler = _saved['target_scaler']
        print("   🔒 Frozen scaler mode — reusing saved feature/target scalers.")
    else:
        if freeze_scaler:
            print("   ⚠️  freeze_scaler requested but no saved scaler found — fitting fresh.")
        feature_scaler = StandardScaler()
        # Fit on all 4 zones so HYDRO/DE features stay in a moderate scaled range
        # (-0.56 for zero-lag zones). Fitting on DK1+DK2 only pushes HYDRO/DE to
        # -0.99, amplifying the noise they inject through message passing.
        feature_scaler.fit(np.vstack([x_dk1[:train_idx_limit], x_dk2[:train_idx_limit], x_de[:train_idx_limit], x_hydro[:train_idx_limit]]))

        target_scaler = StandardScaler()
        target_scaler.fit(np.hstack([y_dk1[:train_idx_limit], y_dk2[:train_idx_limit]]).reshape(-1, 1))
    
    # 5. Build PyTorch Geometric HeteroData Object
    data = HeteroData()
    
    # Scale inputs and assign node representations
    x_all = np.vstack([
        feature_scaler.transform(x_dk1),
        feature_scaler.transform(x_dk2),
        feature_scaler.transform(x_hydro),
        feature_scaler.transform(x_de)
    ])
    y_all = np.hstack([y_dk1, y_dk2, y_hydro, y_de])
    
    data['hour'].x = torch.tensor(x_all, dtype=torch.float32)
    data['hour'].y = torch.tensor(y_all, dtype=torch.float32) # Stored raw for exact cash error evaluation

    # Set up localized Market Node static profiling matrix (4 Zones)
    # Mapping index sequence: 0: DK1, 1: DK2, 2: HYDRO, 3: DE
    data['market'].x = torch.eye(4, dtype=torch.float32)
    
    # 6. Establish Structural Graph Relationships (Edges)
    belongs_src = []
    belongs_dst = []
    for zone_idx in range(4):
        start_offset = zone_idx * num_hours
        belongs_src.extend(list(range(start_offset, start_offset + num_hours)))
        belongs_dst.extend([zone_idx] * num_hours)
        
    data['hour', 'belongs_to', 'market'].edge_index = torch.tensor([belongs_src, belongs_dst], dtype=torch.long)
    data['market', 'rev_belongs_to', 'hour'].edge_index = torch.tensor([belongs_dst, belongs_src], dtype=torch.long)
    
    # Day-Ahead Autoregressive Lag Edges.
    # Connect each hour to the same hour 1 day (t-24), 2 days (t-48), and 1 week
    # (t-168) back — all known at gate closure, matching the day-ahead setup.
    DAY_AHEAD_LAGS = [24, 48, 168]
    lag_src_parts, lag_dst_parts = [], []
    for zone_idx in range(4):
        offset = zone_idx * num_hours
        for lag in DAY_AHEAD_LAGS:
            t = np.arange(lag, num_hours)
            lag_src_parts.append(offset + (t - lag))
            lag_dst_parts.append(offset + t)
    lag_src = np.concatenate(lag_src_parts)
    lag_dst = np.concatenate(lag_dst_parts)

    data['hour', 'lag_to', 'hour'].edge_index = torch.tensor(np.stack([lag_src, lag_dst]), dtype=torch.long)

    # Same-Timestep Cross-Zone Edges (physical interconnect topology).
    # Node offsets: DK1=[0,T), DK2=[T,2T), HYDRO(SE3)=[2T,3T), DE=[3T,4T)
    #
    # We connect ONLY full-coverage zones at the hour level:
    #   DK1 ↔ DK2  : Great Belt HVDC (~1000 MW)
    #   DK2 ↔ HYDRO: Øresund link to SE3 (~1700 MW)
    # DE is deliberately EXCLUDED from hour-level co_occurs_with edges: its
    # price series ends 2024-12-31 and is forward-filled with a stale constant
    # across 2025 — which is exactly the val/test window. Wiring DE directly to
    # DK1/DK2 here would inject that stale value straight into the evaluation
    # predictions. DE still participates more coarsely via the market node,
    # where the model can learn to down-weight it.
    t_all  = np.arange(num_hours)
    dk1, dk2, hyd = 0, num_hours, 2 * num_hours
    co_src = np.concatenate([
        dk1 + t_all, dk2 + t_all,   # DK1→DK2, DK2→DK1
        dk2 + t_all, hyd + t_all,   # DK2→HYDRO, HYDRO→DK2
    ])
    co_dst = np.concatenate([
        dk2 + t_all, dk1 + t_all,
        hyd + t_all, dk2 + t_all,
    ])
    data['hour', 'co_occurs_with', 'hour'].edge_index = torch.tensor(
        np.stack([co_src, co_dst]), dtype=torch.long
    )

    # Cross-Border Spatial Grid Interconnects (Market-to-Market Topology)
    # 0: DK1, 1: DK2, 2: HYDRO(SE3), 3: DE  — all physical cross-border links
    inter_src = [0, 1, 0, 3, 0, 2, 1, 2, 1, 3]
    inter_dst = [1, 0, 3, 0, 2, 0, 2, 1, 3, 1]

    # Physical capacity weights (MW): Great Belt, Kontek, DK1-HYDRO,
    # Øresund, Kriegers Flak/Energybridge
    inter_weights = [1000.0, 1000.0, 2000.0, 1500.0, 600.0, 600.0,
                     1700.0, 1700.0, 600.0, 600.0]
    
    data['market', 'interconnects', 'market'].edge_index = torch.tensor([inter_src, inter_dst], dtype=torch.long)
    data['market', 'interconnects', 'market'].edge_attr = torch.tensor(inter_weights, dtype=torch.float32).view(-1, 1)
    
    # 7. Construct Non-Overlapping Spatiotemporal Validation Masks
    train_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    val_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    test_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    
    val_idx_limit = int(num_hours * 0.9)
    
    for zone_idx in range(4):
        offset = zone_idx * num_hours
        train_mask[offset : offset + train_idx_limit] = True
        val_mask[offset + train_idx_limit : offset + val_idx_limit] = True
        test_mask[offset + val_idx_limit : offset + num_hours] = True
        
    data['hour'].train_mask = train_mask
    data['hour'].val_mask = val_mask
    data['hour'].test_mask = test_mask
    
    # Keep track of individual zone sequence dimensions for our readout layer splits
    data['hour'].num_hours_per_zone = num_hours
    
    # Save elements out
    torch.save(data, GRAPH_DIR / "hetero_graph.pt")
    with open(GRAPH_DIR / "hetero_scalers.pkl", "wb") as f:
        pickle.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, f)
        
    print(f"✅ Success! HeteroData artifact package compiled. Graph nodes: {data['hour'].x.shape[0]} entities.")

if __name__ == "__main__":
    freeze = "--freeze-scaler" in sys.argv
    build_heterogeneous_spatiotemporal_graph(freeze_scaler=freeze)