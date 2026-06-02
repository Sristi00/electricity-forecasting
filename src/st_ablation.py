"""
Ablation study for the ST-HeteroSAGE model (best checkpoint: 151.1 DKK).

Inference-time variants — no retraining. Each variant disables one structural
component by either monkey-patching the model or swapping out edge indices.

Variants
--------
A  Full model              — 151.1 DKK baseline
B  No TCN                  — temporal CausalTCN bypassed (spatial only)
C  No HeteroConv           — spatial conv bypassed (temporal only, no zone coupling)
D  No co_occurs_with       — DK1<->DK2 and DK2<->HYDRO direct edges removed
E  No market bridge        — belongs_to / rev_belongs_to / interconnects removed
F  No DK2<->HYDRO          — revert to DK1<->DK2 coupling only (no SE3)

Results saved to artifacts_hetero/st_ablation_results.json
"""
import sys, json, types, copy
import numpy as np
import torch
import torch.nn.functional as F
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
_DUMMY = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)


# ── Setup ─────────────────────────────────────────────────────────────────────

def _setup():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    T = int(data['hour'].num_hours_per_zone)
    n2 = 2 * T

    # Masks (DK1+DK2 only for primary evaluation)
    dk12 = torch.zeros(4 * T, dtype=torch.bool, device=DEVICE); dk12[:n2] = True
    tr_mask   = data['hour'].train_mask.to(DEVICE) & dk12
    test_mask = data['hour'].test_mask .to(DEVICE) & dk12

    # Target scaler — identical to st_train.py
    y_raw = data['hour'].y.cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(y_raw[tr_mask.cpu().numpy()].reshape(-1, 1))

    # Load model
    ckpt = ARTIFACTS_HETERO_DIR / "best_st_hetero_model.pt"
    model = HeteroSTPriceForecaster(
        in_channels=data['hour'].x.shape[1], hidden_channels=128, num_st_blocks=2,
        temporal_dilations=(1, 4, 24), temporal_kernel=7,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    model.eval()

    x_dict = {'hour': data['hour'].x, 'market': data['market'].x}
    base_ei = {et: data[et].edge_index for et in SPATIAL_EDGE_TYPES}

    return model, x_dict, base_ei, y_raw, scaler, test_mask, T


# ── Evaluation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _eval(model, x_dict, ei, y_raw, scaler, test_mask):
    out_scaled = model(x_dict, ei).cpu().numpy()
    tm = test_mask.cpu().numpy()
    y_pred = scaler.inverse_transform(out_scaled[tm].reshape(-1, 1)).flatten()
    y_true = y_raw[tm]
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── Ablation patches ──────────────────────────────────────────────────────────

def _patch_no_tcn(model):
    """Return orig forwards; patch all STBlock.forward to skip temporal step."""
    saved = []
    for block in model.st_blocks:
        saved.append(block.forward)
        def _fwd(self, x_dict, ei):
            h = self.spatial_conv(x_dict, ei)
            h_hour   = F.relu(self.bn_hour_sp  (h['hour']))   + x_dict['hour']
            h_market = F.relu(self.bn_market_sp(h['market'])) + x_dict['market']
            return {'hour': h_hour, 'market': h_market}
        block.forward = types.MethodType(_fwd, block)
    return saved


def _patch_no_spatial(model):
    """Patch all STBlock.forward to skip HeteroConv (TCN only)."""
    saved = []
    for block in model.st_blocks:
        saved.append(block.forward)
        def _fwd(self, x_dict, ei):
            T_  = x_dict['hour'].size(0) // 4
            H_  = x_dict['hour'].size(1)
            h_zones = x_dict['hour'].view(4, T_, H_).permute(0, 2, 1)
            h_zones = self.tcn(h_zones)
            h_hour  = h_zones.permute(0, 2, 1).contiguous().view(4 * T_, H_)
            h_hour  = F.relu(self.bn_temporal(h_hour)) + x_dict['hour']
            return {'hour': h_hour, 'market': x_dict['market']}
        block.forward = types.MethodType(_fwd, block)
    return saved


def _restore(model, saved):
    for block, fwd in zip(model.st_blocks, saved):
        block.forward = fwd


def _ei_no_cooccurs(base_ei):
    ei = dict(base_ei)
    ei[('hour', 'co_occurs_with', 'hour')] = _DUMMY
    return ei


def _ei_no_market(base_ei):
    ei = dict(base_ei)
    for et in [('hour','belongs_to','market'),
               ('market','rev_belongs_to','hour'),
               ('market','interconnects','market')]:
        ei[et] = _DUMMY
    return ei


def _ei_no_hydro(base_ei, T):
    """Remove DK2<->HYDRO pairs, keep only DK1<->DK2 in co_occurs_with."""
    co = base_ei[('hour', 'co_occurs_with', 'hour')]
    mask = (co[0] < 2*T) & (co[1] < 2*T)   # both in DK1 or DK2 range
    filtered = co[:, mask]
    ei = dict(base_ei)
    ei[('hour', 'co_occurs_with', 'hour')] = filtered if filtered.shape[1] > 0 else _DUMMY
    return ei


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n🔬 ST-HeteroSAGE Ablation Study")
    model, x_dict, base_ei, y_raw, scaler, test_mask, T = _setup()

    results = {}

    # A — full model
    m = _eval(model, x_dict, base_ei, y_raw, scaler, test_mask)
    results['A_full'] = m
    print(f"  A  Full model           MAE={m['mae']:.2f}  R²={m['r2']:.4f}")

    # B — no TCN
    saved = _patch_no_tcn(model)
    m = _eval(model, x_dict, base_ei, y_raw, scaler, test_mask)
    results['B_no_tcn'] = m
    _restore(model, saved)
    delta = m['mae'] - results['A_full']['mae']
    print(f"  B  No TCN (temporal)    MAE={m['mae']:.2f}  Δ={delta:+.2f}")

    # C — no HeteroConv (spatial bypassed)
    saved = _patch_no_spatial(model)
    m = _eval(model, x_dict, base_ei, y_raw, scaler, test_mask)
    results['C_no_spatial'] = m
    _restore(model, saved)
    delta = m['mae'] - results['A_full']['mae']
    print(f"  C  No spatial HeteroConv MAE={m['mae']:.2f}  Δ={delta:+.2f}")

    # D — no co_occurs_with
    m = _eval(model, x_dict, _ei_no_cooccurs(base_ei), y_raw, scaler, test_mask)
    results['D_no_cooccurs'] = m
    delta = m['mae'] - results['A_full']['mae']
    print(f"  D  No co_occurs_with    MAE={m['mae']:.2f}  Δ={delta:+.2f}")

    # E — no market bridge
    m = _eval(model, x_dict, _ei_no_market(base_ei), y_raw, scaler, test_mask)
    results['E_no_market'] = m
    delta = m['mae'] - results['A_full']['mae']
    print(f"  E  No market bridge     MAE={m['mae']:.2f}  Δ={delta:+.2f}")

    # F — no DK2<->HYDRO
    m = _eval(model, x_dict, _ei_no_hydro(base_ei, T), y_raw, scaler, test_mask)
    results['F_no_hydro'] = m
    delta = m['mae'] - results['A_full']['mae']
    print(f"  F  No DK2<->HYDRO       MAE={m['mae']:.2f}  Δ={delta:+.2f}")

    out = ARTIFACTS_HETERO_DIR / "st_ablation_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out}")
    return results


if __name__ == "__main__":
    run()
