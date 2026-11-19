"""G-TCN with switchable components for the revision ablations.

With the defaults (decay="memory", adj="learn", fusion="attention", branches="both") the
module is identical to TCNmaster/TCN/word_cnn/GCNtcnbat.py: same layers, same parameter
names, same initialisation, same forward pass. Checkpoints of the original model load into it.

Variants used in the revision:
  decay    = "memory" | "none" (Abl-1) | "all" (Abl-2: decay also over the 7 lead positions)
  adj      = "learn" | "identity" | "uniform" (Abl-3: fixed matrices)
             | "null"     (B3: extra sentinel source in the row softmax, so a row may receive ~0 input)
             | "selfloop" (B4: self-loop fixed at 0.5, off-diagonal weights row-softmax * 0.5)
  fusion   = "attention" | "sum" (Abl-4: 0.5*(x+y)) | "concat" (Abl-4b: Linear(2,1) on [x, y])
  branches = "both" | "gcn" (graph branch only, as the standalone GCN) | "tcn" (temporal branch only)
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from TCNmaster.TCN.tcn import TemporalConvNet  # noqa: E402


class GraphConvolution(nn.Module):
    """Batched GCN layer (identical to GCNtcnbat.py)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).repeat(input.size(0), 1, 1)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            output += self.bias
        return output


class LearnableAdj(nn.Module):
    """Adjacency of the variable graph. mode="learn" is identical to GCNtcnbat.py."""

    def __init__(self, num_vars=6, mode="learn"):
        super().__init__()
        self.num_vars, self.mode = num_vars, mode
        if mode in ("learn", "selfloop"):
            self.raw_adj = nn.Parameter(torch.eye(num_vars) + 0.01 * torch.randn(num_vars, num_vars))
        elif mode == "null":
            # last column is the sentinel ("no information") source; initialised like the off-diagonal
            raw = torch.cat([torch.eye(num_vars), torch.zeros(num_vars, 1)], dim=1)
            self.raw_adj = nn.Parameter(raw + 0.01 * torch.randn(num_vars, num_vars + 1))
        elif mode == "identity":
            self.register_buffer("fixed", torch.eye(num_vars))
        elif mode == "uniform":
            self.register_buffer("fixed", torch.full((num_vars, num_vars), 1.0 / num_vars))
        else:
            raise ValueError(mode)

    def forward(self, x=None):
        if self.mode == "learn":
            return F.softmax(self.raw_adj, dim=-1)
        if self.mode == "null":
            return F.softmax(self.raw_adj, dim=-1)[:, :self.num_vars]  # rows sum to <= 1
        if self.mode == "selfloop":
            eye = torch.eye(self.num_vars, device=self.raw_adj.device)
            off = self.raw_adj.masked_fill(eye.bool(), float("-inf"))
            return 0.5 * eye + 0.5 * F.softmax(off, dim=-1)
        return self.fixed


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.3, emb_dropout=0.1,
                 decay="memory", adj="learn", fusion="attention", branches="both"):
        super().__init__()
        self.decay, self.fusion, self.branches = decay, fusion, branches
        if branches in ("both", "gcn"):
            self.learnable_adj = LearnableAdj(num_vars=6, mode=adj)
            self.gc1 = GraphConvolution(22, input_size)
            self.gc2 = GraphConvolution(input_size, 22)
        if branches in ("both", "tcn"):
            self.encoder = nn.Linear(6, input_size)
            self.norm1 = nn.LayerNorm(22)        # unused in forward (kept to mirror GCNtcnbat.py)
            self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
            self.decoder = nn.Linear(num_channels[-1], output_size)
        if branches == "both":
            self.linear1 = nn.Linear(2, 1)       # used only by fusion="concat"
            self.att_x = nn.Linear(1, 1)
            self.att_y = nn.Linear(1, 1)
        self.drop = nn.Dropout(emb_dropout)
        if branches in ("both", "gcn") and decay != "none":
            self.time_decay = nn.Parameter(torch.tensor(0.1))
        self.memory_days = 15
        if branches in ("both", "tcn"):
            self.encoder.weight.data.normal_(0, 0.01)
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.normal_(0, 0.01)

    def _decayed(self, input):
        if self.decay == "none":
            return input
        T = input.shape[1]
        span = self.memory_days if self.decay == "memory" else T
        t = torch.arange(span, device=input.device).float()
        w = torch.exp(-self.time_decay * (span - 1 - t))
        w = (w / w.sum()).view(1, span, 1)
        out = input.clone()
        out[:, :span, :] *= w
        return out

    def forward(self, input):
        """input: (B, T, C), T = 22 (15 memory + 7 forecast days)."""
        if self.branches in ("both", "tcn"):
            emb = self.drop(self.encoder(input)).transpose(1, 2)
            enc = self.tcn(emb).transpose(1, 2)
            y = self.decoder(enc)[:, :, -1].unsqueeze(-1)
            if self.branches == "tcn":
                return y
        x = self._decayed(input).transpose(1, 2)
        adj = self.learnable_adj(x)
        x = self.drop(F.relu(self.gc1(x, adj)))
        x = self.gc2(x, adj).transpose(1, 2)[:, :, -1].unsqueeze(-1)
        if self.branches == "gcn":
            return x
        if self.fusion == "attention":
            att = torch.softmax(torch.cat([self.att_x(x), self.att_y(y)], dim=-1), dim=-1)
            return att[..., 0:1] * x + att[..., 1:2] * y
        if self.fusion == "sum":
            return 0.5 * (x + y)
        if self.fusion == "concat":
            return self.linear1(torch.cat([x, y], dim=-1))
        raise ValueError(self.fusion)


VARIANTS = {
    "full": {},
    "abl1_no_decay": {"decay": "none"},
    "abl2_decay_all": {"decay": "all"},
    "abl3_adj_identity": {"adj": "identity"},
    "abl3_adj_uniform": {"adj": "uniform"},
    "abl4_fusion_sum": {"fusion": "sum"},
    "abl5_gcn_only": {"branches": "gcn"},
    "abl6_tcn_only": {"branches": "tcn"},
    "b3_null_node": {"adj": "null"},
    "b4_selfloop": {"adj": "selfloop"},
}


def build(variant, d_model=64, n_blocks=4):
    return TCN(input_size=d_model, output_size=6, num_channels=[d_model] * n_blocks, **VARIANTS[variant])
