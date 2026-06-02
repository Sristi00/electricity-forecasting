"""
Interpretability analysis for the ST-HeteroSAGE model.

Three analyses:
  1. Gradient-based feature importance
     For each test DK1/DK2 node, compute |∂MSE/∂x_i| averaged over the test
     window. Reports which of the 17 input features drive predictions most.

  2. Error pattern by time
     MAE bucketed by hour-of-day (0–23) and day-of-week (0–6) from the 2025
     test timestamps. Reveals systematic weaknesses (peak hours, weekends).

  3. TCN temporal sensitivity
     For a representative batch of test nodes, compute gradient of output w.r.t.
     each of the 17 feature dimensions aggregated across the 4-zone TCN input.
     Shows which feature channels the temporal pathway is most sensitive to.

Outputs: artifacts_hetero/st_interpretability.json
"""
import sys, json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR as ARTIFACTS_HETERO_DIR
from hetero_st_model import HeteroSTPriceForecaster
from hetero_pipeline import prepare_multi_area_data

SPATIAL_EDGE_TYPES = [
    ('hour', 'co_occurs_with', 'hour'),
    ('hour', 'belongs_to',     'market'),
    ('market', 'rev_belongs_to', 'hour'),
    ('market', 'interconnects',  'market'),
]

FEATURE_NAMES = [
    'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
    'price_roll_mean_24h', 'price_roll_std_24h',
    'temperature_c', 'wind_speed_ms', 'cloud_cover_pct', 'humidity_pct',
    'load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]


def _setup():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    T = int(data['hour'].num_hours_per_zone)
    dk12 = torch.zeros(4 * T, dtype=torch.bool); dk12[:2*T] = True
    tr_mask   = data['hour'].train_mask & dk12
    test_mask = data['hour'].test_mask  & dk12

    y_raw = data['hour'].y.numpy()
    scaler = StandardScaler()
    scaler.fit(y_raw[tr_mask.numpy()].reshape(-1, 1))

    model = HeteroSTPriceForecaster(
        in_channels=17, hidden_channels=128, num_st_blocks=2,
        temporal_dilations=(1, 4, 24), temporal_kernel=7,
    ).to(DEVICE)
    model.load_state_dict(torch.load(
        ARTIFACTS_HETERO_DIR / "best_st_hetero_model.pt",
        map_location=DEVICE, weights_only=False))
    model.eval()

    ei = {et: data[et].edge_index.to(DEVICE) for et in SPATIAL_EDGE_TYPES}
    return model, data, ei, y_raw, scaler, test_mask, T


# ── 1. Gradient-based feature importance ─────────────────────────────────────

def feature_importance(model, data, ei, y_raw, scaler, test_mask):
    print("  [1/3] Gradient-based feature importance...")
    # detach+clone avoids mutating data['hour'].x in-place
    x_hour = data['hour'].x.detach().clone().to(DEVICE).requires_grad_(True)
    x_market = data['market'].x.to(DEVICE)
    x_dict = {'hour': x_hour, 'market': x_market}

    out = model(x_dict, ei)
    y_scaled = torch.tensor(
        scaler.transform(y_raw.reshape(-1, 1)).flatten(), dtype=torch.float32, device=DEVICE
    )
    loss = torch.nn.functional.mse_loss(out[test_mask.to(DEVICE)], y_scaled[test_mask.to(DEVICE)])
    loss.backward()

    # Average |gradient| over all DK1+DK2 test nodes
    grad = x_hour.grad.detach().cpu().numpy()
    tm = test_mask.numpy()
    importance = np.abs(grad[tm]).mean(axis=0)  # shape [17]

    ranked = sorted(zip(FEATURE_NAMES, importance.tolist()), key=lambda x: -x[1])
    print("     Top 10 features by |∂loss/∂x|:")
    for name, val in ranked[:10]:
        print(f"       {name:<25} {val:.6f}")
    return [{"feature": n, "importance": v} for n, v in ranked]


# ── 2. Error pattern by time ──────────────────────────────────────────────────

def error_by_time(model, data, ei, y_raw, scaler, test_mask, T):
    print("  [2/3] Error pattern by hour-of-day and day-of-week...")
    with torch.no_grad():
        x_dict = {'hour': data['hour'].x.to(DEVICE), 'market': data['market'].x.to(DEVICE)}
        out_scaled = model(x_dict, ei).cpu().numpy()

    tm = test_mask.numpy()
    y_pred = scaler.inverse_transform(out_scaled[tm].reshape(-1, 1)).flatten()
    y_true = y_raw[tm]
    abs_err = np.abs(y_pred - y_true)

    # Timestamps from pipeline
    df_dk1, *_ = prepare_multi_area_data()
    ts = df_dk1['timestamp'].values   # shape [T]

    # Test slice: last 10% of T hours, interleaved across DK1+DK2
    n_test_per_zone = test_mask[:T].numpy().sum()
    # Test window is the last n_test_per_zone hours in DK1 (same timestamps for DK2)
    ts_test = ts[-n_test_per_zone:]

    import pandas as pd
    ts_pd = pd.to_datetime(ts_test)
    hour_mae, dow_mae = {}, {}
    for h in range(24):
        mask_h = ts_pd.hour == h
        if mask_h.any():
            # abs_err has 2*n_test (DK1 + DK2 concatenated in test order)
            dk1_err = abs_err[:n_test_per_zone][mask_h]
            dk2_err = abs_err[n_test_per_zone:][mask_h]
            hour_mae[h] = float(np.concatenate([dk1_err, dk2_err]).mean())
    for d in range(7):
        mask_d = ts_pd.dayofweek == d
        if mask_d.any():
            dk1_err = abs_err[:n_test_per_zone][mask_d]
            dk2_err = abs_err[n_test_per_zone:][mask_d]
            dow_mae[d] = float(np.concatenate([dk1_err, dk2_err]).mean())

    day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    print(f"     Worst hour:  {max(hour_mae, key=hour_mae.get):02d}:00  MAE={max(hour_mae.values()):.1f} DKK")
    print(f"     Best hour:   {min(hour_mae, key=hour_mae.get):02d}:00  MAE={min(hour_mae.values()):.1f} DKK")
    print(f"     Worst day:   {day_names[max(dow_mae, key=dow_mae.get)]}  MAE={max(dow_mae.values()):.1f} DKK")
    print(f"     Best day:    {day_names[min(dow_mae, key=dow_mae.get)]}  MAE={min(dow_mae.values()):.1f} DKK")
    return {"hour_of_day": hour_mae, "day_of_week": {day_names[k]: v for k, v in dow_mae.items()}}


# ── 3. TCN temporal sensitivity ───────────────────────────────────────────────

def tcn_sensitivity(model, data, ei, y_raw, scaler, test_mask):
    print("  [3/3] TCN temporal sensitivity by feature channel...")
    x_hour = data['hour'].x.detach().clone().to(DEVICE).requires_grad_(True)
    x_dict = {'hour': x_hour, 'market': data['market'].x.to(DEVICE)}
    out = model(x_dict, ei)
    y_scaled = torch.tensor(
        scaler.transform(y_raw.reshape(-1, 1)).flatten(), dtype=torch.float32, device=DEVICE
    )
    loss = torch.nn.functional.mse_loss(out[test_mask.to(DEVICE)], y_scaled[test_mask.to(DEVICE)])
    loss.backward()

    grad = x_hour.grad.detach().cpu().numpy()
    # Average over test-zone nodes (DK1+DK2 test window)
    tm = test_mask.numpy()
    channel_sens = np.abs(grad[tm]).mean(axis=0)  # [17]
    ranked = sorted(zip(FEATURE_NAMES, channel_sens.tolist()), key=lambda x: -x[1])
    print("     Top 5 channels:")
    for name, val in ranked[:5]:
        print(f"       {name:<25} {val:.6f}")
    return [{"feature": n, "sensitivity": v} for n, v in ranked]


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n🔍 ST-HeteroSAGE Interpretability")
    model, data, ei, y_raw, scaler, test_mask, T = _setup()

    fi   = feature_importance(model, data, ei, y_raw, scaler, test_mask)
    errt = error_by_time(model, data, ei, y_raw, scaler, test_mask, T)
    tcns = tcn_sensitivity(model, data, ei, y_raw, scaler, test_mask)

    results = {
        "feature_importance":    fi,
        "error_by_time":         errt,
        "tcn_channel_sensitivity": tcns,
    }
    out = ARTIFACTS_HETERO_DIR / "st_interpretability.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out}")
    return results


if __name__ == "__main__":
    run()
