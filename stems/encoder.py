"""
Spatial-Temporal Encoder (Eq 12-15).

Architecture:
    SpatialGCN          – 3-layer GCN, 64 hidden units (Eq 12)
    TemporalTransformer – multi-head self-attention over T=24-step window (Eq 13-14)
    STEncoder           – fuses spatial + temporal into 64-dim representation (Eq 15)

torch_geometric is used when available; otherwise FallbackGCNConv provides an
equivalent manual implementation with symmetric normalisation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Optional torch_geometric import
# --------------------------------------------------------------------------
_PYGEOM_AVAILABLE = False
try:
    from torch_geometric.nn import GCNConv  # type: ignore
    _PYGEOM_AVAILABLE = True
except ImportError:
    pass


# --------------------------------------------------------------------------
# FallbackGCNConv – pure PyTorch GCN layer (symmetric normalisation)
# --------------------------------------------------------------------------

class FallbackGCNConv(nn.Module):
    r"""Manual GCN convolution following Kipf & Welling (2017).

    Computes:  H' = D̂^{-1/2} Â D̂^{-1/2} H W
    where Â = A + I (self-loops added).
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, in_channels)
        adj : (B, B) – row-normalised adjacency (self-loops already removed)
        """
        B = adj.size(0)
        device = adj.device
        # Add self-loops: Â = A + I
        adj_hat = adj + torch.eye(B, device=device)

        # Symmetric normalisation: D̂^{-1/2} Â D̂^{-1/2}
        deg = adj_hat.sum(dim=1)
        d_inv_sqrt = torch.diag(deg.pow(-0.5))
        adj_norm = d_inv_sqrt @ adj_hat @ d_inv_sqrt

        # H' = Â_norm * H * W
        out = adj_norm @ x @ self.weight
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


# --------------------------------------------------------------------------
# SpatialGCN (Eq 12)
# --------------------------------------------------------------------------

class SpatialGCN(nn.Module):
    r"""Three-layer GCN that aggregates spatial context across buildings.

    Eq 12:  H^{(l+1)} = σ(D̂^{-1/2} Â D̂^{-1/2} H^{(l)} W^{(l)})
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        dims = [in_channels] + [hidden_dim] * num_layers

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(FallbackGCNConv(dims[i], dims[i + 1]))

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, in_channels)
        adj : (B, B) normalised adjacency

        Returns
        -------
        h : (B, hidden_dim)
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h   # (B, hidden_dim)


# --------------------------------------------------------------------------
# TemporalTransformer (Eq 13-14)
# --------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    r"""Multi-head self-attention over a sliding window of T=24 timesteps.

    Eq 13 (attention):
        e_{τ,τ'} = (Q_τ K_{τ'}ᵀ) / √d_k

    Eq 14 (output):
        z_i = LayerNorm(x_i + Σ_{τ'} softmax(e_{τ,τ'}) V_{τ'})
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        window_size: int = 24,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        # Project raw observations into embed_dim
        self.input_proj = nn.Linear(obs_dim, embed_dim)

        # Positional encoding (sinusoidal, fixed)
        self.register_buffer("pos_enc", self._build_pos_enc(window_size, embed_dim))

        # Multi-head self-attention (Eq 13-14)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    # ------------------------------------------------------------------
    @staticmethod
    def _build_pos_enc(window: int, dim: int) -> torch.Tensor:
        """Fixed sinusoidal positional encoding, shape (1, window, dim)."""
        pe = torch.zeros(window, dim)
        pos = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
        return pe.unsqueeze(0)   # (1, window, dim)

    # ------------------------------------------------------------------
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        history : (B, T, obs_dim)  – rolling observation window

        Returns
        -------
        z : (B, embed_dim)  – temporal context vector (last timestep output)
        """
        B, T, _ = history.shape
        # Project and add positional encoding
        x = self.input_proj(history) + self.pos_enc[:, :T, :]  # (B, T, embed_dim)

        # Self-attention (Eq 13-14)
        attn_out, _ = self.attn(x, x, x)                       # (B, T, embed_dim)
        z = self.norm(x + attn_out)                             # residual + LN

        return z[:, -1, :]   # take the last-step representation  (B, embed_dim)


# --------------------------------------------------------------------------
# STEncoder – fusion layer (Eq 15)
# --------------------------------------------------------------------------

class STEncoder(nn.Module):
    r"""Fuses spatial GCN output h_i and temporal Transformer output z_i.

    Eq 15:  r_i = W_s h_i + W_t z_i + b
    """

    def __init__(
        self,
        obs_dim: int,
        spatial_dim: int = 64,
        temporal_dim: int = 32,
        output_dim: int = 64,
        gcn_num_layers: int = 3,
        num_heads: int = 4,
        window_size: int = 24,
    ) -> None:
        super().__init__()

        self.spatial_gcn = SpatialGCN(
            in_channels=obs_dim,
            hidden_dim=spatial_dim,
            num_layers=gcn_num_layers,
        )

        self.temporal_transformer = TemporalTransformer(
            obs_dim=obs_dim,
            embed_dim=temporal_dim,
            num_heads=num_heads,
            window_size=window_size,
        )

        # Fusion projection (Eq 15)
        self.W_s = nn.Linear(spatial_dim, output_dim, bias=False)
        self.W_t = nn.Linear(temporal_dim, output_dim, bias=True)
        self.out_dim = output_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, obs_dim)          current observations
        adj     : (B, B)                normalised adjacency matrix
        history : (B, T, obs_dim)       rolling observation history

        Returns
        -------
        r : (B, output_dim)             spatial-temporal representations
        """
        h = self.spatial_gcn(x, adj)              # (B, spatial_dim)
        z = self.temporal_transformer(history)    # (B, temporal_dim)
        r = self.W_s(h) + self.W_t(z)            # (B, output_dim) – Eq 15
        return r

    # ------------------------------------------------------------------
    def batch_forward(
        self,
        x_nb: torch.Tensor,
        adj: torch.Tensor,
        history_nb: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised encoding for a batch of N graph instances.

        Each of the N instances has B buildings sharing the same adjacency
        structure (adj).  Processes the Temporal Transformer on all N*B
        sequences in one call, then runs the GCN on each of the N instances.

        Parameters
        ----------
        x_nb       : (N, B, obs_dim)
        adj        : (B, B)
        history_nb : (N, B, T, obs_dim)

        Returns
        -------
        r_nb : (N, B, output_dim)
        """
        N, B, obs_dim = x_nb.shape
        T = history_nb.shape[2]

        # ---- Temporal Transformer: process all N*B sequences at once ----
        hist_flat = history_nb.view(N * B, T, obs_dim)   # (N*B, T, obs_dim)
        z_flat = self.temporal_transformer(hist_flat)     # (N*B, temporal_dim)
        z_nb = z_flat.view(N, B, -1)                      # (N, B, temporal_dim)

        # ---- Spatial GCN: process each of N instances (shared adj) ----
        # For efficiency we process N GCN calls; GCN is cheap (small B).
        h_nb = torch.zeros(N, B, self.spatial_gcn.out_dim, device=x_nb.device)
        for n in range(N):
            h_nb[n] = self.spatial_gcn(x_nb[n], adj)     # (B, spatial_dim)

        # ---- Fusion (Eq 15) ----
        r_nb = self.W_s(h_nb) + self.W_t(z_nb)           # (N, B, output_dim)
        return r_nb
