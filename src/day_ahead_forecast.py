"""
Multi-area day-ahead forecasting analysis.

Addresses the gaps in the original evaluation:
  1. Per-zone metrics for all four areas  (DK1, DK2, HYDRO, DE)
  2. Hourly and weekly price profiles     (avg predicted vs actual by time slot)
  3. MAE by hour-of-day position          (which delivery hours are hardest)
     — the model is genuine day-ahead: it predicts all 24 hours of day D directly
       from lags known at gate closure (>=24h). There is no recursive feedback,
       so we report error per delivery hour rather than a compounding 1-step sim.

Works with both legacy (9-feature) and current (13-feature) checkpoints.
Outputs saved to artifacts_hetero/day_ahead_results.json
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR, load_y_scaler, inverse_scale_y
from hetero_models import load_hetero_model
from hetero_pipeline import prepare_multi_area_data

ZONE_NAMES = ['DK1', 'DK2', 'HYDRO', 'DE']


def _load():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)
    model, x_override = load_hetero_model(data, ARTIFACTS_DIR / "best_hetero_model.pt", DEVICE)
    x_base = (x_override.cpu() if x_override is not None else data['hour'].x).clone()
    y_mean, y_scale = load_y_scaler()
    return model, data, num_hours, x_base, y_mean, y_scale


@torch.no_grad()
def _infer(model, data, x_base, num_hours, y_mean=0.0, y_scale=1.0):
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    x_dict['hour'] = x_base.to(DEVICE)
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    out_scaled = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()
    return inverse_scale_y(out_scaled, y_mean, y_scale)


def compute_per_zone_metrics(model, data, num_hours, x_base, y_mean=0.0, y_scale=1.0):
    out = _infer(model, data, x_base, num_hours, y_mean, y_scale)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    results = {}
    for z, name in enumerate(ZONE_NAMES):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zm = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]
        smape = float(np.mean(2 * np.abs(p_z - y_z) / (np.abs(y_z) + np.abs(p_z) + 1e-8)) * 100)
        mape  = float(np.mean(np.abs((p_z - y_z) / (np.abs(y_z) + 1e-8))) * 100)
        results[name] = {
            'mae':   round(float(mean_absolute_error(y_z, p_z)), 4),
            'rmse':  round(float(np.sqrt(np.mean((p_z - y_z)**2))), 4),
            'r2':    round(float(r2_score(y_z, p_z)), 4),
            'smape': round(smape, 4),
            'mape':  round(mape, 4),
        }
    return results


def compute_hourly_price_profiles(model, data, num_hours, x_base, y_mean=0.0, y_scale=1.0):
    """Average predicted vs actual price by hour-of-day and day-of-week."""
    import pandas as pd
    df_dk1, _, _, _, _ = prepare_multi_area_data()
    ts = pd.to_datetime(df_dk1['timestamp'].values)
    test_start  = int(num_hours * 0.9)
    hour_of_day = ts[test_start:].hour.values
    day_of_week = ts[test_start:].dayofweek.values

    out = _infer(model, data, x_base, num_hours, y_mean, y_scale)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    profiles = {}
    for z_idx, z_name in enumerate(['DK1', 'DK2']):
        sl = slice(z_idx * num_hours, (z_idx + 1) * num_hours)
        zm = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]

        by_hour = {
            str(h): {
                'actual':    round(float(y_z[hour_of_day == h].mean()), 2) if (hour_of_day == h).any() else 0.0,
                'predicted': round(float(p_z[hour_of_day == h].mean()), 2) if (hour_of_day == h).any() else 0.0,
            }
            for h in range(24)
        }
        by_dow = {
            str(d): {
                'actual':    round(float(y_z[day_of_week == d].mean()), 2) if (day_of_week == d).any() else 0.0,
                'predicted': round(float(p_z[day_of_week == d].mean()), 2) if (day_of_week == d).any() else 0.0,
            }
            for d in range(7)
        }
        profiles[z_name] = {'by_hour': by_hour, 'by_day_of_week': by_dow}
    return profiles


def compute_hour_position_mae(model, data, num_hours, x_base, y_mean=0.0, y_scale=1.0):
    """
    Day-ahead error by delivery hour (00:00 … 23:00).

    The model emits all 24 hours of day D in a single forward pass from lags known
    at gate closure — there is no recursive feedback to compound. The meaningful
    horizon question is therefore: are some delivery hours (e.g. evening peak)
    intrinsically harder to forecast than others?

    Returns a per-zone list of 24 MAE values indexed by hour-of-day.
    """
    import pandas as pd
    df_dk1, _, _, _, _ = prepare_multi_area_data()
    ts = pd.to_datetime(df_dk1['timestamp'].values)
    hour_of_day_full = ts.hour.values  # aligned to the per-zone hour index

    out = _infer(model, data, x_base, num_hours, y_mean, y_scale)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    horizon_mae = {name: [] for name in ZONE_NAMES}
    for z, name in enumerate(ZONE_NAMES):
        sl  = slice(z * num_hours, (z + 1) * num_hours)
        zm  = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]
        hod_z = hour_of_day_full[zm]
        for h in range(24):
            sel = hod_z == h
            mae = float(mean_absolute_error(y_z[sel], p_z[sel])) if sel.any() else 0.0
            horizon_mae[name].append(round(mae, 4))
    return horizon_mae


def run_day_ahead_analysis():
    print("\n📅 Running Day-Ahead Multi-Zone Forecast Analysis...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data, num_hours, x_base, y_mean, y_scale = _load()
    n_feats = x_base.shape[1]
    print(f"   Model input features: {n_feats}")

    print("  [1/3] Per-zone metrics (DK1, DK2, HYDRO, DE)...")
    zone_metrics = compute_per_zone_metrics(model, data, num_hours, x_base, y_mean, y_scale)

    print("  [2/3] Hourly and weekly price profiles...")
    profiles = compute_hourly_price_profiles(model, data, num_hours, x_base, y_mean, y_scale)

    print("  [3/3] MAE by delivery hour (day-ahead, direct 24h)...")
    horizon = compute_hour_position_mae(model, data, num_hours, x_base, y_mean, y_scale)

    summary = {
        'per_zone_metrics': zone_metrics,
        'price_profiles':   profiles,
        'horizon_mae':      horizon,
    }

    out_path = ARTIFACTS_DIR / "day_ahead_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── readable output ───────────────────────────────────────────────────────
    print("\n📊 Per-Zone Test Metrics (note: HYDRO & DE have zero prices in current DB):")
    print(f"  {'Zone':<8} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'SMAPE':>8} {'Data'}")
    print("  " + "-" * 54)
    data_status = {'DK1': 'real', 'DK2': 'real', 'HYDRO': 'zero-fill', 'DE': 'zero-fill'}
    for name, m in zone_metrics.items():
        flag = data_status.get(name, '?')
        print(f"  {name:<8} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>7.4f} {m['smape']:>7.2f}%  {flag}")

    print("\n🕓 DK1 MAE by delivery hour (day-ahead, all 24h predicted at once):")
    dk1_h = horizon.get('DK1', [])
    for h in range(0, 24, 3):
        if h < len(dk1_h):
            print(f"    {h:02d}h → {dk1_h[h]:.1f} DKK")

    print("\n🕐 DK1 Daily Price Profile (avg by hour):")
    for h in range(0, 24, 3):
        slot = profiles.get('DK1', {}).get('by_hour', {}).get(str(h), {})
        a, p = slot.get('actual', 0), slot.get('predicted', 0)
        print(f"    {h:02d}h: actual={a:.0f}  predicted={p:.0f} DKK")

    print(f"\n✅ Saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_day_ahead_analysis()
