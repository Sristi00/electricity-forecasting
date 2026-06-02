"""
Interpretability analysis for the heterogeneous GNN.

Produces three analyses:
  1. Gradient-based feature importance — which input features drive each zone's price
  2. Error pattern analysis — MAE by hour-of-day and day-of-week
  3. Cross-zone prediction correlation — how zone forecasts co-move

Uses load_hetero_model() so it works with both legacy (9-feature) and
current (13-feature) checkpoints automatically.

Outputs saved to artifacts_hetero/interpretability_summary.json
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR, load_y_scaler, inverse_scale_y
from hetero_models import load_hetero_model
from hetero_pipeline import prepare_multi_area_data

FEATURE_NAMES_9  = ['lag_24h', 'lag_48h', 'lag_168h', 'roll_24h_mean', 'roll_24h_std',
                    'temperature', 'wind_speed', 'cloud_cover', 'humidity']
FEATURE_NAMES_13 = FEATURE_NAMES_9 + ['hour_sin', 'hour_cos', 'week_sin', 'week_cos']
ZONE_NAMES = ['DK1', 'DK2', 'HYDRO', 'DE']


def _load():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)
    model, x_override = load_hetero_model(data, ARTIFACTS_DIR / "best_hetero_model.pt", DEVICE)
    y_mean, y_scale = load_y_scaler()
    return model, data, num_hours, x_override, y_mean, y_scale


def _build_x_dict(data, x_override):
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    if x_override is not None:
        x_dict['hour'] = x_override.to(DEVICE)
    return x_dict


def compute_feature_importance(model, data, num_hours, x_override):
    """Gradient saliency: average |∂output/∂input_hour| over test nodes per zone."""
    x_dict = _build_x_dict(data, x_override)
    x_hour_leaf = x_dict['hour'].clone().requires_grad_(True)
    x_dict['hour'] = x_hour_leaf
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    out = model(x_dict, ei, num_hours=num_hours).view(-1)
    test_mask = data['hour'].test_mask.to(DEVICE)
    out[test_mask].sum().backward()

    grads = x_hour_leaf.grad.detach().cpu().numpy()
    n_feats = grads.shape[1]
    feat_names = FEATURE_NAMES_13[:n_feats]

    test_mask_np = data['hour'].test_mask.cpu().numpy()
    results = {}
    for z, name in enumerate(ZONE_NAMES):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zone_grads = np.abs(grads[sl][test_mask_np[sl]])
        importance = zone_grads.mean(axis=0)
        importance = importance / (importance.max() + 1e-8)
        results[name] = {f: float(s) for f, s in zip(feat_names, importance)}
    return results


def compute_error_analysis(model, data, num_hours, x_override, y_mean=0.0, y_scale=1.0):
    """MAE grouped by hour-of-day and day-of-week for DK1 and DK2."""
    import pandas as pd
    x_dict = _build_x_dict(data, x_override)
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        out_scaled = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()
    out = inverse_scale_y(out_scaled, y_mean, y_scale)

    y = data['hour'].y.cpu().numpy()
    test_mask_np = data['hour'].test_mask.cpu().numpy()

    df_dk1, _, _, _, _ = prepare_multi_area_data()
    timestamps = pd.to_datetime(df_dk1['timestamp'].values)
    test_start = int(num_hours * 0.9)
    test_ts    = timestamps[test_start:]
    hour_of_day = test_ts.hour.values
    day_of_week = test_ts.dayofweek.values

    by_hour, by_dow = {}, {}
    for z_idx, z_name in enumerate(['DK1', 'DK2']):
        sl = slice(z_idx * num_hours, (z_idx + 1) * num_hours)
        zm = test_mask_np[sl]
        ae = np.abs(y[sl][zm] - out[sl][zm])

        by_hour[z_name] = {
            str(h): float(ae[hour_of_day == h].mean()) if (hour_of_day == h).any() else 0.0
            for h in range(24)
        }
        by_dow[z_name] = {
            str(d): float(ae[day_of_week == d].mean()) if (day_of_week == d).any() else 0.0
            for d in range(7)
        }
    return by_hour, by_dow


def compute_zone_correlations(model, data, num_hours, x_override):
    """Pearson correlation of test-set predictions across all four zones."""
    x_dict = _build_x_dict(data, x_override)
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        out = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()

    y = data['hour'].y.cpu().numpy()
    tm = data['hour'].test_mask.cpu().numpy()

    pred_zones, actual_zones = [], []
    for z in range(4):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zm = tm[sl]
        pred_zones.append(out[sl][zm])
        actual_zones.append(y[sl][zm])

    def corr_matrix(arrays):
        mat = np.stack(arrays, axis=1)
        return np.corrcoef(mat.T).tolist()

    return {
        'pred_correlation':   corr_matrix(pred_zones),
        'actual_correlation': corr_matrix(actual_zones),
        'zone_order': ZONE_NAMES,
    }


def run_interpretability():
    print("\n🔍 Running Interpretability Analysis...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data, num_hours, x_override, y_mean, y_scale = _load()
    feat_mode = "legacy (9-feature)" if x_override is not None else "current (13-feature)"
    print(f"   Loaded model: {feat_mode}")

    print("  [1/3] Gradient-based feature importance...")
    feat_imp = compute_feature_importance(model, data, num_hours, x_override)

    print("  [2/3] Error pattern analysis (hour-of-day, day-of-week)...")
    by_hour, by_dow = compute_error_analysis(model, data, num_hours, x_override, y_mean, y_scale)

    print("  [3/3] Cross-zone prediction correlations...")
    zone_corr = compute_zone_correlations(model, data, num_hours, x_override)

    summary = {
        'feature_importance': feat_imp,
        'mae_by_hour_of_day': by_hour,
        'mae_by_day_of_week': by_dow,
        'zone_correlations':  zone_corr,
    }

    out_path = ARTIFACTS_DIR / "interpretability_summary.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n📊 Top-3 Features by Zone (gradient importance, normalised 0-1):")
    for zone, scores in feat_imp.items():
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        row  = "  |  ".join(f"{n}: {v:.3f}" for n, v in top3)
        print(f"  {zone:<6}: {row}")

    print("\n🕐 DK1 MAE by Hour of Day (selected hours):")
    dk1_hr = by_hour.get('DK1', {})
    for h in range(0, 24, 3):
        print(f"    {h:02d}h → {dk1_hr.get(str(h), 0):.1f} DKK")

    print("\n📅 DK1 MAE by Day of Week:")
    dk1_dow = by_dow.get('DK1', {})
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for d, name in enumerate(days):
        print(f"    {name}: {dk1_dow.get(str(d), 0):.1f} DKK")

    print(f"\n✅ Saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_interpretability()
