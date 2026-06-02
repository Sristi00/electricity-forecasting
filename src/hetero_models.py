# OVERWRITE EXACTLY: src/hetero_models.py
"""
Heterogeneous GNN models for multi-area electricity price forecasting.
Includes:
  - LegacyHeteroPriceForecaster  : original trained model (9 features, shared output)
  - HeteroPriceForecaster        : upgraded model (13 features, zone-specific heads)
  - HeteroGATPriceForecaster     : GAT variant with attention weight extraction
  - load_hetero_model()          : auto-detecting checkpoint loader
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear, BatchNorm

class LegacyHeteroPriceForecaster(torch.nn.Module):
    """
    Original trained architecture (checkpoint: best_hetero_model.pt).
    9 input features, shared out_projection output head.
    Kept for backwards compatibility with existing checkpoints.
    """
    def __init__(self, metadata, hour_in_features=9, hidden_channels=128):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.market_lin = Linear(4, hidden_channels)
        self.hour_lin   = Linear(hour_in_features, hidden_channels)
        self.conv1 = HeteroConv({
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'belongs_to', 'market'):      SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('market', 'rev_belongs_to', 'hour'):  SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'):            SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        self.conv2 = HeteroConv({
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('market', 'rev_belongs_to', 'hour'):  SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'):            SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        self.out_projection = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, num_hours=None):
        h = {
            'market': F.leaky_relu(self.market_lin(x_dict['market']), 0.2),
            'hour':   F.leaky_relu(self.hour_lin(x_dict['hour']), 0.2),
        }
        h1 = self.conv1(h, edge_index_dict)
        h['market'] = F.leaky_relu(h1['market'] + h['market'], 0.2)
        h['hour']   = F.leaky_relu(h1['hour']   + h['hour'],   0.2)
        h2 = self.conv2(h, edge_index_dict)
        h['market'] = F.leaky_relu(h2['market'] + h['market'], 0.2)
        h['hour']   = F.leaky_relu(h2['hour']   + h['hour'],   0.2)
        return self.out_projection(h['hour'])


class HeteroPriceForecaster(torch.nn.Module):
    """
    3-layer HeteroSAGE with:
      - co_occurs_with edges (DK1↔DK2, DK2↔HYDRO, DK1↔DE, DK2↔DE)
      - 48h lag edge (in addition to 24h and 168h)
      - belongs_to included in all 3 layers (market stays updated)
      - BatchNorm on hour embeddings after each conv layer
      - 2-layer market MLP for richer zone-embedding capacity
      - Single shared output head (vs 4 zone-specific heads):
          eliminates gradient noise from DE/HYDRO zero-filled targets;
          100 % of the regression signal targets DK1/DK2 accuracy.
    """
    def __init__(self, metadata, hour_in_features, hidden_channels=128):
        super().__init__()
        self.hidden_channels = hidden_channels

        # 2-layer market projection — richer zone embedding than a plain Linear
        self.market_lin  = Linear(4, hidden_channels)
        self.market_lin2 = Linear(hidden_channels, hidden_channels)
        self.hour_lin    = Linear(hour_in_features, hidden_channels)

        _edge_types_full = {
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'belongs_to', 'market'):      SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('market', 'rev_belongs_to', 'hour'):  SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'):            SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'co_occurs_with', 'hour'):    SAGEConv(hidden_channels, hidden_channels),
        }
        _edge_types_late = {
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'belongs_to', 'market'):      SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('market', 'rev_belongs_to', 'hour'):  SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'):            SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'co_occurs_with', 'hour'):    SAGEConv(hidden_channels, hidden_channels),
        }

        self.conv1 = HeteroConv(_edge_types_full,  aggr='sum')
        self.conv2 = HeteroConv(_edge_types_late,  aggr='sum')
        self.conv3 = HeteroConv({
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('market', 'rev_belongs_to', 'hour'):  SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'):            SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'co_occurs_with', 'hour'):    SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')

        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        # Single shared output head — one regression head for all zones.
        # Previously 4 zone-specific heads; DE/HYDRO heads injected gradient
        # noise from zero-filled targets. Shared head means 100% of regression
        # signal drives DK1/DK2 accuracy (loss is masked to DK1+DK2 only).
        self.output_head = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.LeakyReLU(0.2),
            Linear(hidden_channels // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, num_hours=None):
        # No dropout: BatchNorm already regularises; BN+Dropout interact adversely.
        m0 = F.leaky_relu(self.market_lin(x_dict['market']), 0.2)
        h = {
            'market': F.leaky_relu(self.market_lin2(m0) + m0, 0.2),  # 2-layer residual MLP
            'hour':   F.leaky_relu(self.hour_lin(x_dict['hour']), 0.2),
        }

        h1 = self.conv1(h, edge_index_dict)
        h['market'] = F.leaky_relu(h1['market'] + h['market'], 0.2)
        h['hour']   = F.leaky_relu(self.bn1(h1['hour']) + h['hour'], 0.2)

        h2 = self.conv2(h, edge_index_dict)
        h['market'] = F.leaky_relu(h2['market'] + h['market'], 0.2)
        h['hour']   = F.leaky_relu(self.bn2(h2['hour']) + h['hour'], 0.2)

        h3 = self.conv3(h, edge_index_dict)
        h['market'] = F.leaky_relu(h3.get('market', h['market']) + h['market'], 0.2)
        h['hour']   = F.leaky_relu(self.bn3(h3['hour']) + h['hour'], 0.2)

        # Single shared head — applied to all 4T hour nodes; loss is externally
        # masked to DK1+DK2 for the primary objective.
        return self.output_head(h['hour'])


class HeteroGATPriceForecaster(torch.nn.Module):
    """
    Heterogeneous GAT model with per-edge-type attention and optional attention
    weight extraction for interpretability analysis.

    Node layout (hour): DK1[0:N], DK2[N:2N], HYDRO[2N:3N], DE[3N:4N]
    """
    def __init__(self, metadata, hour_in_features, hidden_channels=128, heads=4):
        super().__init__()
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        self.hidden_channels = hidden_channels
        self.heads = heads
        per_head = hidden_channels // heads

        self.market_lin = Linear(4, hidden_channels)
        self.hour_lin = Linear(hour_in_features, hidden_channels)

        # Layer 1 — one GATConv per edge type
        self.gat1_mm = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False, dropout=0.1)
        self.gat1_hm = GATConv((hidden_channels, hidden_channels), per_head,
                                heads=heads, concat=True, add_self_loops=False, dropout=0.1)
        self.gat1_mh = GATConv((hidden_channels, hidden_channels), per_head,
                                heads=heads, concat=True, add_self_loops=False, dropout=0.1)
        self.gat1_hh = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False, dropout=0.1)
        self.gat1_co = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False, dropout=0.1)

        # Layer 2
        self.gat2_mm = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False)
        self.gat2_mh = GATConv((hidden_channels, hidden_channels), per_head,
                                heads=heads, concat=True, add_self_loops=False)
        self.gat2_hh = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False)
        self.gat2_co = GATConv(hidden_channels, per_head, heads=heads, concat=True,
                                add_self_loops=False)

        self.head_dk1   = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_dk2   = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_de    = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_hydro = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))

    def forward(self, x_dict, edge_index_dict, num_hours=None, return_attention=False):
        h_m = F.leaky_relu(self.market_lin(x_dict['market']), 0.2)
        h_h = F.leaky_relu(self.hour_lin(x_dict['hour']), 0.2)

        ei = edge_index_dict
        attn = {}

        if return_attention:
            h_mm, (_, w_mm) = self.gat1_mm(h_m, ei[('market', 'interconnects', 'market')],
                                             return_attention_weights=True)
            h_hm, (_, w_hm) = self.gat1_hm((h_h, h_m), ei[('hour', 'belongs_to', 'market')],
                                             return_attention_weights=True)
            h_mh, (_, w_mh) = self.gat1_mh((h_m, h_h), ei[('market', 'rev_belongs_to', 'hour')],
                                             return_attention_weights=True)
            h_hh, (_, w_hh) = self.gat1_hh(h_h, ei[('hour', 'lag_to', 'hour')],
                                             return_attention_weights=True)
            h_co = self.gat1_co(h_h, ei[('hour', 'co_occurs_with', 'hour')])
            attn = {
                'market_interconnects': w_mm.detach().cpu(),
                'hour_belongs_to_market': w_hm.detach().cpu(),
                'market_to_hour': w_mh.detach().cpu(),
                'temporal_lag': w_hh.detach().cpu(),
            }
        else:
            h_mm = self.gat1_mm(h_m, ei[('market', 'interconnects', 'market')])
            h_hm = self.gat1_hm((h_h, h_m), ei[('hour', 'belongs_to', 'market')])
            h_mh = self.gat1_mh((h_m, h_h), ei[('market', 'rev_belongs_to', 'hour')])
            h_hh = self.gat1_hh(h_h, ei[('hour', 'lag_to', 'hour')])
            h_co = self.gat1_co(h_h, ei[('hour', 'co_occurs_with', 'hour')])

        h_m = F.leaky_relu(h_mm + h_hm + h_m, 0.2)
        h_h = F.leaky_relu(h_mh + h_hh + h_co + h_h, 0.2)

        h_mm2 = self.gat2_mm(h_m, ei[('market', 'interconnects', 'market')])
        h_mh2 = self.gat2_mh((h_m, h_h), ei[('market', 'rev_belongs_to', 'hour')])
        h_hh2 = self.gat2_hh(h_h, ei[('hour', 'lag_to', 'hour')])
        h_co2 = self.gat2_co(h_h, ei[('hour', 'co_occurs_with', 'hour')])

        h_m = F.leaky_relu(h_mm2 + h_m, 0.2)
        h_h = F.leaky_relu(h_mh2 + h_hh2 + h_co2 + h_h, 0.2)

        h_dk1   = h_h[0:num_hours]
        h_dk2   = h_h[num_hours:2 * num_hours]
        h_hydro = h_h[2 * num_hours:3 * num_hours]
        h_de    = h_h[3 * num_hours:4 * num_hours]

        out = torch.cat([self.head_dk1(h_dk1), self.head_dk2(h_dk2),
                         self.head_hydro(h_hydro), self.head_de(h_de)], dim=0)

        if return_attention:
            return out, attn
        return out


def load_hetero_model(data, model_path, device):
    """
    Auto-detect checkpoint architecture and return (model, x_hour_override).

    Legacy checkpoints (hour_lin weight shape [128, 9], key 'out_projection') are
    loaded into LegacyHeteroPriceForecaster; the caller should use x_hour_override
    (first 9 columns of hour features) instead of the full 13-column tensor.

    Current checkpoints (13 features, zone-specific heads) load into
    HeteroPriceForecaster; x_hour_override is None (use graph features as-is).
    """
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # Infer hidden_channels from the first linear layer shape
    hidden_channels = checkpoint['market_lin.weight'].shape[0]
    in_feats = checkpoint['hour_lin.weight'].shape[1]

    if 'out_projection.weight' in checkpoint:
        model = LegacyHeteroPriceForecaster(
            metadata=data.metadata(), hour_in_features=in_feats,
            hidden_channels=hidden_channels,
        ).to(device)
        x_hour_override = data['hour'].x[:, :in_feats]
    else:
        # Both old (zone-specific heads) and new (shared head) checkpoints use
        # HeteroPriceForecaster — load_state_dict handles the key mapping.
        model = HeteroPriceForecaster(
            metadata=data.metadata(), hour_in_features=in_feats,
            hidden_channels=hidden_channels,
        ).to(device)
        x_hour_override = None

    model.load_state_dict(checkpoint)
    model.eval()
    return model, x_hour_override