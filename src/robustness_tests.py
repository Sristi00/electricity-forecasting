"""
Robustness evaluation for the heterogeneous GNN.

Three perturbation scenarios applied to ALL node features to simulate
real-world data-quality degradation across the full sensor network:

  1. Gaussian noise injection   — σ ∈ {5%, 10%, 20%, 30%} of each feature std
  2. Feature dropout            — randomly zero-out features at {10%, 20%, 30%}
  3. Price-spike simulation     — amplify lag columns by {2×, 3×, 5×} for 5% of nodes

Works with both legacy (9-feature) and current (13-feature) checkpoints via
load_hetero_model().

Results saved to artifacts_hetero/robustness_results.json
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

ZONE_NAMES = ['DK1', 'DK2', 'HYDRO', 'DE']
LAG_COLS = [0, 1, 2, 3, 4]   # lag_24h, lag_48h, lag_168h, roll_24h_mean, roll_24h_std


def _load():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)
    model, x_override = load_hetero_model(data, ARTIFACTS_DIR / "best_hetero_model.pt", DEVICE)
    x_base = (x_override.cpu() if x_override is not None else data['hour'].x).clone()
    y_mean, y_scale = load_y_scaler()
    return model, data, num_hours, x_base, y_mean, y_scale


@torch.no_grad()
def _eval(model, data, x_hour, num_hours, y_mean=0.0, y_scale=1.0):
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    x_dict['hour'] = x_hour.to(DEVICE)
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    out_scaled = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()
    out = inverse_scale_y(out_scaled, y_mean, y_scale)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    metrics = {}
    for z, name in enumerate(ZONE_NAMES):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zm = tm[sl]
        mae  = float(mean_absolute_error(y[sl][zm], out[sl][zm]))
        rmse = float(np.sqrt(np.mean((out[sl][zm] - y[sl][zm]) ** 2)))
        r2   = float(r2_score(y[sl][zm], out[sl][zm]))
        metrics[name] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}
    return metrics


def _pct(base, new):
    return round(100.0 * (new - base) / (base + 1e-8), 2)


def _add_delta(results, baseline):
    return {
        zone: {**v, 'mae_delta_pct': _pct(baseline[zone]['mae'], v['mae'])}
        for zone, v in results.items()
    }


def test_gaussian_noise(model, data, num_hours, baseline, x_base, y_mean=0.0, y_scale=1.0,
                        noise_levels=(0.05, 0.10, 0.20, 0.30)):
    feat_std = x_base.std(dim=0).clamp(min=1e-6)
    results = {}
    for sigma in noise_levels:
        x_n = x_base + sigma * feat_std * torch.randn_like(x_base)
        results[f'noise_{int(sigma*100)}pct'] = _add_delta(_eval(model, data, x_n, num_hours, y_mean, y_scale), baseline)
    return results


def test_feature_dropout(model, data, num_hours, baseline, x_base, y_mean=0.0, y_scale=1.0,
                         rates=(0.10, 0.20, 0.30)):
    rng = np.random.default_rng(42)
    results = {}
    for rate in rates:
        mask = torch.tensor(rng.random(x_base.shape) > rate, dtype=torch.float32)
        results[f'dropout_{int(rate*100)}pct'] = _add_delta(_eval(model, data, x_base * mask, num_hours, y_mean, y_scale), baseline)
    return results


def test_price_spike(model, data, num_hours, baseline, x_base, y_mean=0.0, y_scale=1.0,
                     multipliers=(2.0, 3.0, 5.0)):
    rng = np.random.default_rng(99)
    spike_idx = np.where(rng.random(len(x_base)) < 0.05)[0]
    results = {}
    for mult in multipliers:
        x_s = x_base.clone()
        x_s[spike_idx, :][:, LAG_COLS] *= mult
        results[f'spike_{int(mult)}x'] = _add_delta(_eval(model, data, x_s, num_hours, y_mean, y_scale), baseline)
    return results


def run_robustness_tests():
    print("\n🛡️  Running Robustness Test Suite...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data, num_hours, x_base, y_mean, y_scale = _load()
    n_feats = x_base.shape[1]
    print(f"   Model input features: {n_feats}")

    print("  [0/3] Baseline (clean)...")
    baseline = _eval(model, data, x_base, num_hours, y_mean, y_scale)

    print("  [1/3] Gaussian noise injection...")
    noise_r = test_gaussian_noise(model, data, num_hours, baseline, x_base, y_mean, y_scale)

    print("  [2/3] Feature dropout...")
    drop_r = test_feature_dropout(model, data, num_hours, baseline, x_base, y_mean, y_scale)

    print("  [3/3] Price spike simulation...")
    spike_r = test_price_spike(model, data, num_hours, baseline, x_base, y_mean, y_scale)

    summary = {
        'baseline': baseline,
        'gaussian_noise': noise_r,
        'feature_dropout': drop_r,
        'price_spike': spike_r,
    }

    out_path = ARTIFACTS_DIR / "robustness_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── readable output ───────────────────────────────────────────────────────
    b_dk1 = baseline['DK1']['mae']
    print(f"\n📊 Robustness Summary — DK1 MAE vs clean baseline ({b_dk1:.1f} DKK):")

    for group, rows in [('Gaussian noise', noise_r), ('Feature dropout', drop_r), ('Price spike', spike_r)]:
        print(f"\n  {group}:")
        for k, v in rows.items():
            delta = v['DK1']['mae_delta_pct']
            print(f"    {k:<22}: {v['DK1']['mae']:.1f} DKK  ({delta:+.1f}%)")

    print(f"\n✅ Saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_robustness_tests()
