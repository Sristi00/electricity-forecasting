"""
Heterogeneous Spatio-Temporal GNN for day-ahead electricity price forecasting.

Architecture: N × (spatial HeteroConv → temporal CausalTCN) blocks.

Spatial sub-block (HeteroConv)
  Handles same-timestep cross-zone dependencies using the heterogeneous graph:
    hour --co_occurs_with--> hour   (DK1 ↔ DK2 physical interconnect)
    hour --belongs_to-------> market
    market --rev_belongs_to-> hour
    market --interconnects---> market

Temporal sub-block (CausalTCN)
  Applied per zone along the time axis after each spatial step.
  Replaces the lag_to graph edges with a dilated causal convolution stack,
  giving a receptive field of ~175 hours (7 days) per block.
  Causal: only leftward (past) padding, so no look-ahead leakage.

Node layout in hetero_graph.pt (zone-blocked):
  indices [0,       T)   → DK1
  indices [T,      2T)   → DK2
  indices [2T,     3T)   → HYDRO (SE3)
  indices [3T,     4T)   → DE
where T = num_hours_per_zone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class CausalTCN(nn.Module):
    """Stack of dilated causal 1-D convolutions for temporal modeling.

    Each layer adds a residual: h = ReLU(BN(Conv(x))) + x.
    Causal: pads (kernel-1)*dilation frames on the left only, then trims
    right to keep the output length equal to the input length.

    Receptive field per layer = (kernel_size - 1) * dilation.
    Default (kernel=7, dilations=(1,4,24)): cumulative RF = 6+24+144 = 174 h.
    """

    def __init__(self, channels: int, dilations=(1, 4, 24), kernel_size: int = 7):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size,
                          padding=(kernel_size - 1) * d, dilation=d)
            )
            # BatchNorm1d on [zones, channels, T] normalises per-channel
            self.bns.append(nn.BatchNorm1d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_zones, channels, T]
        T = x.size(2)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x)[:, :, :T]   # trim right (causal)
            h = F.relu(bn(h)) + x   # residual
            x = h
        return x


class STBlock(nn.Module):
    """One spatio-temporal block: spatial HeteroConv then zone-wise CausalTCN."""

    def __init__(self, hidden_channels: int, temporal_dilations=(1, 4, 24),
                 temporal_kernel: int = 7):
        super().__init__()
        H = hidden_channels

        # Spatial: same-timestep heterogeneous message passing
        # lag_to edges are excluded — temporal deps handled by TCN below.
        self.spatial_conv = HeteroConv({
            ('hour',   'co_occurs_with', 'hour'):   SAGEConv(H, H),
            ('hour',   'belongs_to',     'market'): SAGEConv(H, H),
            ('market', 'rev_belongs_to', 'hour'):   SAGEConv(H, H),
            ('market', 'interconnects',  'market'): SAGEConv(H, H),
        }, aggr='mean')

        self.bn_hour_sp   = nn.BatchNorm1d(H)
        self.bn_market_sp = nn.BatchNorm1d(H)

        # Temporal: per-zone causal TCN
        self.tcn        = CausalTCN(H, temporal_dilations, temporal_kernel)
        self.bn_temporal = nn.BatchNorm1d(H)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        num_hour_nodes = x_dict['hour'].size(0)
        T = num_hour_nodes // 4
        H = x_dict['hour'].size(1)

        # ── Spatial ──────────────────────────────────────────────────────────
        h = self.spatial_conv(x_dict, edge_index_dict)
        h_hour   = F.relu(self.bn_hour_sp  (h['hour']))   + x_dict['hour']
        h_market = F.relu(self.bn_market_sp(h['market'])) + x_dict['market']

        # ── Temporal (hour nodes only) ────────────────────────────────────────
        # [4T, H] → [4, H, T] → TCN → [4, H, T] → [4T, H]
        h_zones = h_hour.view(4, T, H).permute(0, 2, 1)   # [4, H, T]
        h_zones = self.tcn(h_zones)                        # [4, H, T]
        h_hour_t = h_zones.permute(0, 2, 1).contiguous().view(4 * T, H)
        h_hour_t = F.relu(self.bn_temporal(h_hour_t)) + h_hour  # residual

        return {'hour': h_hour_t, 'market': h_market}


class HeteroSTPriceForecaster(nn.Module):
    """Heterogeneous Spatio-Temporal Price Forecaster.

    Args:
        in_channels:        Input feature size for 'hour' nodes (17 after pipeline).
        hidden_channels:    Internal hidden size (default 128).
        num_st_blocks:      Number of ST blocks stacked (default 2).
        temporal_dilations: Dilation schedule for the CausalTCN in each block.
        temporal_kernel:    Kernel size for the CausalTCN.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 num_st_blocks: int = 2,
                 temporal_dilations=(1, 4, 24), temporal_kernel: int = 7):
        super().__init__()
        H = hidden_channels

        # Input projections (per node type)
        self.hour_proj    = nn.Linear(in_channels, H)
        # 2-layer residual market MLP — richer zone embedding than a plain Linear
        # (matches the HeteroPriceForecaster recipe that lifted hetero to 162.8 DKK).
        self.market_proj  = nn.Linear(4, H)          # 4 = one-hot market size
        self.market_proj2 = nn.Linear(H, H)

        # Spatio-temporal blocks
        self.st_blocks = nn.ModuleList([
            STBlock(H, temporal_dilations, temporal_kernel)
            for _ in range(num_st_blocks)
        ])

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        m0 = F.relu(self.market_proj(x_dict['market']))
        x = {
            'hour':   F.relu(self.hour_proj(x_dict['hour'])),
            'market': F.relu(self.market_proj2(m0) + m0),   # 2-layer residual MLP
        }
        for block in self.st_blocks:
            x = block(x, edge_index_dict)
        return self.fc(x['hour']).squeeze(-1)
