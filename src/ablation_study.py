"""
Ablation study: contribution of each graph component to forecast accuracy.

Inference-time ablation variants (no retraining needed):
  A. Full model          — baseline
  B. No temporal edges   — remove 'lag_to' (hour→hour autoregressive links)
  C. No spatial edges    — remove 'interconnects' (cross-border market connections)
  D. No market bridge    — remove 'belongs_to' + 'rev_belongs_to' (market aggregation nodes)
  E. No market context   — B + D combined (graph reduces to isolated zone time-series)
  F. No co_occurs_with   — remove direct DK1↔DK2 same-timestep edges

For each variant, affected edge_index entries are replaced with a single dummy
self-loop (node 0 → node 0) so HeteroConv still receives valid edge tensors
while effectively removing all real connectivity of that type.

Results saved to artifacts_hetero/ablation_results.json
"""
import sys
import json
import copy
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR, load_y_scaler, inverse_scale_y
from hetero_models import load_hetero_model

ZONE_NAMES = ['DK1', 'DK2', 'HYDRO', 'DE']

# Dummy edge: single self-loop at node 0 — keeps HeteroConv happy with 0 real edges
_DUMMY_EDGE = torch.zeros(2, 1, dtype=torch.long)


def _load():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)
    model, x_override = load_hetero_model(data, ARTIFACTS_DIR / "best_hetero_model.pt", DEVICE)
    y_mean, y_scale = load_y_scaler()
    return model, data, num_hours, x_override, y_mean, y_scale


@torch.no_grad()
def _eval(model, data, ei_dict, num_hours, x_override=None, y_mean=0.0, y_scale=1.0):
    """Run inference and return per-zone test metrics."""
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    if x_override is not None:
        x_dict['hour'] = x_override.to(DEVICE)
    ei          = {k: v.to(DEVICE) for k, v in ei_dict.items()}
    out_scaled  = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()
    out         = inverse_scale_y(out_scaled, y_mean, y_scale)
    y           = data['hour'].y.cpu().numpy()
    tm     = data['hour'].test_mask.cpu().numpy()

    zone_metrics = {}
    for z, name in enumerate(ZONE_NAMES):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zm = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]
        mae  = float(mean_absolute_error(y_z, p_z))
        rmse = float(np.sqrt(np.mean((p_z - y_z) ** 2)))
        r2   = float(r2_score(y_z, p_z))
        zone_metrics[name] = {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(r2, 4)}

    # Overall DK1+DK2 combined MAE (primary evaluation zones)
    dk_slices = [tm[0:num_hours], tm[num_hours:2*num_hours]]
    dk_y    = np.concatenate([y[0:num_hours][dk_slices[0]], y[num_hours:2*num_hours][dk_slices[1]]])
    dk_pred = np.concatenate([out[0:num_hours][dk_slices[0]], out[num_hours:2*num_hours][dk_slices[1]]])
    zone_metrics['DK_combined'] = {
        'mae':  round(float(mean_absolute_error(dk_y, dk_pred)), 4),
        'rmse': round(float(np.sqrt(np.mean((dk_pred - dk_y)**2))), 4),
        'r2':   round(float(r2_score(dk_y, dk_pred)), 4),
    }
    return zone_metrics


def run_ablation_study():
    print("\n🔬 Running Ablation Study (inference-time edge removal)...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data, num_hours, x_override, y_mean, y_scale = _load()
    feat_mode = "legacy (9-feature)" if x_override is not None else "current (13-feature)"
    print(f"   Loaded model: {feat_mode}")

    base_ei = {k: v for k, v in data.edge_index_dict.items()}

    results = {}

    print("  [A] Full model (baseline)...")
    results['A_full_model'] = _eval(model, data, base_ei, num_hours, x_override, y_mean, y_scale)

    print("  [B] No temporal edges (remove lag_to)...")
    ei_b = dict(base_ei)
    ei_b[('hour', 'lag_to', 'hour')] = _DUMMY_EDGE.clone()
    results['B_no_temporal'] = _eval(model, data, ei_b, num_hours, x_override, y_mean, y_scale)

    print("  [C] No spatial edges (remove interconnects)...")
    ei_c = dict(base_ei)
    ei_c[('market', 'interconnects', 'market')] = _DUMMY_EDGE.clone()
    results['C_no_spatial'] = _eval(model, data, ei_c, num_hours, x_override, y_mean, y_scale)

    print("  [D] No market aggregation (remove belongs_to + rev_belongs_to)...")
    ei_d = dict(base_ei)
    ei_d[('hour', 'belongs_to', 'market')]     = _DUMMY_EDGE.clone()
    ei_d[('market', 'rev_belongs_to', 'hour')] = _DUMMY_EDGE.clone()
    results['D_no_market_bridge'] = _eval(model, data, ei_d, num_hours, x_override, y_mean, y_scale)

    print("  [E] No market context (spatial + market bridge removed)...")
    ei_e = dict(base_ei)
    ei_e[('market', 'interconnects', 'market')] = _DUMMY_EDGE.clone()
    ei_e[('hour', 'belongs_to', 'market')]       = _DUMMY_EDGE.clone()
    ei_e[('market', 'rev_belongs_to', 'hour')]   = _DUMMY_EDGE.clone()
    results['E_no_market_context'] = _eval(model, data, ei_e, num_hours, x_override, y_mean, y_scale)

    if ('hour', 'co_occurs_with', 'hour') in base_ei:
        print("  [F] No co_occurs_with (remove direct DK1↔DK2 same-timestep edges)...")
        ei_f = dict(base_ei)
        ei_f[('hour', 'co_occurs_with', 'hour')] = _DUMMY_EDGE.clone()
        results['F_no_cooccurs'] = _eval(model, data, ei_f, num_hours, x_override, y_mean, y_scale)

    # ── Compute MAE delta vs full model ──────────────────────────────────────
    base_dk1_mae = results['A_full_model']['DK1']['mae']
    for variant, metrics in results.items():
        for zone in metrics:
            delta = round(metrics[zone]['mae'] - base_dk1_mae, 4)
            results[variant][zone]['mae_vs_full'] = delta

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = ARTIFACTS_DIR / "ablation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Print readable table ─────────────────────────────────────────────────
    print("\n📊 Ablation Results — DK1 MAE (DKK) | Δ vs full model:")
    print(f"\n  {'Variant':<30} {'DK1 MAE':>10} {'Δ MAE':>10} {'DK1 R²':>8}")
    print("  " + "-" * 62)
    variant_labels = {
        'A_full_model':       'A. Full model',
        'B_no_temporal':      'B. No temporal (lag_to)',
        'C_no_spatial':       'C. No spatial (interconnects)',
        'D_no_market_bridge': 'D. No market bridge',
        'E_no_market_context':'E. No market context',
        'F_no_cooccurs':      'F. No co_occurs (DK1↔DK2)',
    }
    for k, label in variant_labels.items():
        if k not in results:
            continue
        m = results[k]['DK1']
        delta_str = f"{m['mae_vs_full']:+.2f}"
        print(f"  {label:<34} {m['mae']:>10.2f} {delta_str:>10} {m['r2']:>8.4f}")

    print("\n  DK_combined MAE:")
    for k, label in variant_labels.items():
        if k not in results:
            continue
        m = results[k]['DK_combined']
        print(f"  {label:<34} {m['mae']:>8.2f} DKK  (R²={m['r2']:.4f})")

    print(f"\n✅ Saved → {out_path}")
    return results


if __name__ == "__main__":
    run_ablation_study()
