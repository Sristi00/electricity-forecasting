# OVERWRITE EXACTLY: src/compare_models.py
"""
Unified project leaderboard — covers all evaluation dimensions:
  1. Baseline comparison (XGBoost / Homogeneous GNN / HeteroSAGE)
  2. Per-zone breakdown (DK1, DK2, HYDRO, DE)
  3. Ablation study summary (edge-type contributions)
  4. Robustness summary (MAE degradation under perturbation)
"""
import json
from pathlib import Path
import pandas as pd


def _load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics():
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent if base_dir.name == 'src' else base_dir

    xgb_m    = _load_json(root_dir / "artifacts" / "xgboost_metrics.json")
    homo_m   = _load_json(base_dir / "artifacts" / "homo_gnn_metrics.json")
    hetero_m = _load_json(root_dir / "src" / "artifacts_hetero" / "hetero_metrics_clean.json")
    da_m     = _load_json(root_dir / "src" / "artifacts_hetero" / "day_ahead_results.json")
    abl_m    = _load_json(root_dir / "src" / "artifacts_hetero" / "ablation_results.json")
    rob_m    = _load_json(root_dir / "src" / "artifacts_hetero" / "robustness_results.json")
    interp_m = _load_json(root_dir / "src" / "artifacts_hetero" / "interpretability_summary.json")

    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  🏆  ELECTRICITY PRICE FORECASTING — FULL EVALUATION REPORT")
    print("═" * 75)

    # ── Section 1: Model Leaderboard ─────────────────────────────────────────
    print("\n▶  SECTION 1: Overall Model Leaderboard (all-zone test set)\n")

    leaderboard = {}
    if xgb_m:
        leaderboard['1. XGBoost (tabular baseline)'] = {
            'MAE (DKK)': round(xgb_m.get('mae', 0), 2),
            'SMAPE':     f"{round(xgb_m.get('smape', 0), 2)}%",
            'R²':        round(xgb_m.get('r2', 0), 4),
        }

    if homo_m:
        leaderboard['2. Homogeneous GNN (GraphSAGE)'] = {
            'MAE (DKK)': round(homo_m.get('mae', 0), 2),
            'SMAPE':     f"{round(homo_m.get('smape', 0), 2)}%",
            'R²':        round(homo_m.get('r2', 0), 4),
        }
    else:
        leaderboard['2. Homogeneous GNN (GraphSAGE)'] = {
            'MAE (DKK)': '(run homo_retrain.py)', 'SMAPE': '—', 'R²': '—',
        }

    if hetero_m:
        leaderboard['3. HeteroSAGE (our model)'] = {
            'MAE (DKK)': round(hetero_m.get('mae', 0), 2),
            'SMAPE':     f"{round(hetero_m.get('smape', 0), 2)}%",
            'R²':        round(hetero_m.get('r2', 0), 4),
        }

    df_lb = pd.DataFrame(leaderboard).T
    print(df_lb.to_string())

    # ── Section 2: Per-Zone Metrics ──────────────────────────────────────────
    print("\n▶  SECTION 2: Per-Zone Metrics — HeteroSAGE\n")
    if da_m:
        zone_rows = {}
        for zone, m in da_m.get('per_zone_metrics', {}).items():
            zone_rows[zone] = {
                'MAE':   m['mae'], 'RMSE':  m['rmse'],
                'R²':    m['r2'],  'SMAPE': f"{m['smape']:.2f}%",
            }
        if zone_rows:
            print(pd.DataFrame(zone_rows).T.to_string())
        else:
            print("  (run day_ahead_forecast.py to generate)")
    else:
        print("  (run day_ahead_forecast.py to generate)")

    # ── Section 3: Ablation Study ────────────────────────────────────────────
    print("\n▶  SECTION 3: Ablation Study — DK1 MAE / R² by edge-type removal\n")
    if abl_m:
        abl_rows = {}
        labels = {
            'A_full_model':       'A. Full model',
            'B_no_temporal':      'B. No temporal (lag_to)',
            'C_no_spatial':       'C. No spatial (interconnects)',
            'D_no_market_bridge': 'D. No market bridge',
            'E_no_market_context':'E. No market context',
        }
        for k, label in labels.items():
            if k in abl_m:
                m = abl_m[k]['DK1']
                abl_rows[label] = {
                    'MAE (DKK)': m['mae'],
                    'Δ vs full': f"{m.get('mae_vs_full', 0):+.2f}",
                    'R²':        m['r2'],
                }
        if abl_rows:
            print(pd.DataFrame(abl_rows).T.to_string())
    else:
        print("  (run ablation_study.py to generate)")

    # ── Section 4: Robustness ────────────────────────────────────────────────
    print("\n▶  SECTION 4: Robustness — DK1 MAE under perturbation\n")
    if rob_m:
        base_mae = rob_m.get('baseline', {}).get('DK1', {}).get('mae', 0)
        print(f"  {'Scenario':<28} {'DK1 MAE':>10} {'Δ%':>8}")
        print("  " + "-" * 50)
        print(f"  {'Baseline (clean)':<28} {base_mae:>10.2f} {'—':>8}")

        for group_key, group_label in [
            ('gaussian_noise',  'Gaussian noise'),
            ('feature_dropout', 'Feature dropout'),
            ('price_spike',     'Price spike'),
        ]:
            for scenario, sv in rob_m.get(group_key, {}).items():
                dk1_v = sv.get('DK1', {})
                label = f"{group_label} / {scenario}"
                mae   = dk1_v.get('mae', 0)
                delta = dk1_v.get('mae_delta_pct', 0)
                print(f"  {label:<28} {mae:>10.2f} {delta:>+7.1f}%")
    else:
        print("  (run robustness_tests.py to generate)")

    # ── Section 5: Interpretability ──────────────────────────────────────────
    print("\n▶  SECTION 5: Top-3 Feature Importance by Zone (gradient saliency)\n")
    if interp_m:
        for zone, scores in interp_m.get('feature_importance', {}).items():
            top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            row  = "  |  ".join(f"{n}: {v:.3f}" for n, v in top3)
            print(f"  {zone:<6}: {row}")
    else:
        print("  (run interpretability.py to generate)")

    # ── Section 6: MAE by delivery hour ─────────────────────────────────────
    print("\n▶  SECTION 6: DK1 MAE by delivery hour (day-ahead, 24h predicted at once)\n")
    if da_m:
        dk1_h = da_m.get('horizon_mae', {}).get('DK1', [])
        if dk1_h:
            print(f"  {'Hour':>10} {'MAE (DKK)':>12}")
            print("  " + "-" * 24)
            for h in [0, 3, 6, 9, 12, 15, 18, 21]:
                if h < len(dk1_h):
                    print(f"  {h:02d}h        {dk1_h[h]:>12.2f}")
        else:
            print("  (run day_ahead_forecast.py to generate)")
    else:
        print("  (run day_ahead_forecast.py to generate)")

    print("\n" + "═" * 75 + "\n")


if __name__ == "__main__":
    load_metrics()
