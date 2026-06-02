"""
Robustness evaluation for the ST-HeteroSAGE model.

Three perturbation scenarios applied to the test-window feature matrix:

  1. Gaussian noise   σ ∈ {5%, 10%, 20%, 30%} of each feature's std
  2. Feature dropout  randomly zero-out features at {10%, 20%, 30%}
  3. Price spike      multiply lag columns (0–4) by {2×, 3×, 5×} for 5% of nodes

All scenarios are inference-only — no retraining.
Baseline is the clean test MAE (151.1 DKK).

Results: artifacts_hetero/st_robustness_results.json
"""
import sys, json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR as ARTIFACTS_HETERO_DIR
from hetero_st_model import HeteroSTPriceForecaster

SPATIAL_EDGE_TYPES = [
    ('hour', 'co_occurs_with', 'hour'),
    ('hour', 'belongs_to',     'market'),
    ('market', 'rev_belongs_to', 'hour'),
    ('market', 'interconnects',  'market'),
]
LAG_COLS = [0, 1, 2, 3, 4]   # price_lag_24h/48h/168h, roll_mean/std

RNG = np.random.default_rng(42)


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

    x_base = data['hour'].x.numpy().copy()   # [4T, 17] in scaled space
    ei = {et: data[et].edge_index.to(DEVICE) for et in SPATIAL_EDGE_TYPES}
    return model, data, ei, x_base, y_raw, scaler, test_mask


@torch.no_grad()
def _eval(model, x_perturbed, data, ei, y_raw, scaler, test_mask):
    x_dict = {
        'hour':   torch.tensor(x_perturbed, dtype=torch.float32, device=DEVICE),
        'market': data['market'].x.to(DEVICE),
    }
    out_scaled = model(x_dict, ei).cpu().numpy()
    tm = test_mask.numpy()
    y_pred = scaler.inverse_transform(out_scaled[tm].reshape(-1, 1)).flatten()
    y_true = y_raw[tm]
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def run():
    print("\n🛡️  ST-HeteroSAGE Robustness Evaluation")
    model, data, ei, x_base, y_raw, scaler, test_mask = _setup()

    results = {}

    # Baseline
    base = _eval(model, x_base, data, ei, y_raw, scaler, test_mask)
    results['baseline'] = base
    print(f"  Baseline              MAE={base['mae']:.2f}  R²={base['r2']:.4f}")

    # Feature std for noise scaling (computed on test nodes)
    tm = test_mask.numpy()
    feat_std = x_base[tm].std(axis=0)   # [17]

    # ── 1. Gaussian noise ─────────────────────────────────────────────────────
    print("\n  [1] Gaussian noise:")
    gn = {}
    for frac in [0.05, 0.10, 0.20, 0.30]:
        x_noisy = x_base.copy()
        noise = RNG.normal(0, frac * feat_std, size=x_base.shape)
        x_noisy += noise
        m = _eval(model, x_noisy, data, ei, y_raw, scaler, test_mask)
        delta = m['mae'] - base['mae']
        gn[f"noise_{int(frac*100)}pct"] = {**m, "delta_mae": delta,
                                            "delta_pct": delta / base['mae'] * 100}
        print(f"    σ={frac*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['gaussian_noise'] = gn

    # ── 2. Feature dropout ────────────────────────────────────────────────────
    print("\n  [2] Feature dropout:")
    fd = {}
    for rate in [0.10, 0.20, 0.30]:
        x_drop = x_base.copy()
        mask = RNG.random(x_drop.shape) < rate
        x_drop[mask] = 0.0
        m = _eval(model, x_drop, data, ei, y_raw, scaler, test_mask)
        delta = m['mae'] - base['mae']
        fd[f"drop_{int(rate*100)}pct"] = {**m, "delta_mae": delta,
                                           "delta_pct": delta / base['mae'] * 100}
        print(f"    rate={rate*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['feature_dropout'] = fd

    # ── 3. Price spike (lag columns amplified for 5% of nodes) ────────────────
    print("\n  [3] Price spike simulation (5% of nodes):")
    ps = {}
    n_spike = max(1, int(0.05 * x_base.shape[0]))
    spike_idx = RNG.choice(x_base.shape[0], size=n_spike, replace=False)
    for factor in [2.0, 3.0, 5.0]:
        x_spike = x_base.copy()
        x_spike[np.ix_(spike_idx, LAG_COLS)] *= factor
        m = _eval(model, x_spike, data, ei, y_raw, scaler, test_mask)
        delta = m['mae'] - base['mae']
        ps[f"spike_{int(factor)}x"] = {**m, "delta_mae": delta,
                                        "delta_pct": delta / base['mae'] * 100}
        print(f"    factor={factor:.0f}×  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['price_spike'] = ps

    out = ARTIFACTS_HETERO_DIR / "st_robustness_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out}")
    return results


if __name__ == "__main__":
    run()
