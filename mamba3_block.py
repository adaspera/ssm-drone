# mamba3_block.py — Mamba-3 2D block for vision (YOLO integration)
# Wraps the Mamba3 SSM layer with 2D spatial handling via VMamba's cross-scan.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("./libs/VMamba")
from vmamba import cross_scan_fn, cross_merge_fn

from mamba3 import Mamba3


class SS2D_Mamba3(nn.Module):
    """2D Selective Scan using Mamba-3 core.
    
    Changes vs SS2D (vmamba_block.py):
    - Uses Mamba3 layer (no conv1d, trapezoidal disc., complex SSM, optional MIMO)
    - Simplified pipeline since Mamba3 already includes gate path internally
    """

    def __init__(
        self,
        dim,
        d_state=16,
        expand=2,
        headdim=64,
        two_way_scan=False,
        # Mamba-3 specific
        use_rope=True,
        use_trapezoidal=True,
        mimo_rank=1,
        bc_bias=True,
        qk_norm=True,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expand)
        self.two_way_scan = two_way_scan
        self.scans = 2 if two_way_scan else 0  # 0=4-way cross, 2=bidirectional

        # Input projection: dim -> d_inner (separate from Mamba3's own projection)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Mamba-3 core (expand=1 since we already expanded)
        self.mamba3 = Mamba3(
            d_model=self.d_inner,
            d_state=d_state,
            expand=1,
            headdim=headdim,
            use_rope=use_rope,
            use_trapezoidal=use_trapezoidal,
            mimo_rank=mimo_rank,
            bc_bias=bc_bias,
            qk_norm=qk_norm,
        )

        # Output projection (applied AFTER merge)
        self.out_proj = nn.Conv2d(self.d_inner, dim, 1, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(dim)

        # Residual scale
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        dtype = x.dtype

        # ── Cross-scan ──
        x_scanned = cross_scan_fn(
            x,
            in_channel_first=True,
            out_channel_first=True,
            scans=self.scans,
            force_torch=False,
        )
        B_s, K, C_s, L = x_scanned.shape  # K=4 or 2, L=H*W

        # Reshape: (B*K, L, C)
        x_seq = x_scanned.view(B_s * K, L, C_s)

        # Input projection  
        proj = self.in_proj(x_seq.to(dtype))
        x_ssm, x_gate = proj.chunk(2, dim=-1)

        # SiLU activation before Mamba-3
        x_ssm = F.silu(x_ssm)

        # Process through Mamba-3
        if x_ssm.device.type != "cpu":
            x_ssm = self.mamba3(x_ssm)

        # Gating
        x_gate = F.silu(x_gate)
        x_out = x_ssm * x_gate  # (B*K, L, d_inner)

        # Reshape for merge: (B, K, d_inner, H, W)
        x_out = x_out.transpose(1, 2).view(B_s, K, self.d_inner, H, W)

        # ── Cross-merge ──
        x_merged = cross_merge_fn(
            x_out,
            in_channel_first=True,
            out_channel_first=True,
            scans=self.scans,
            force_torch=False,
        )

        # Reshape: (B, d_inner, H, W)
        x_merged = x_merged.reshape(B_s, self.d_inner, H, W)

        # Output projection + norm
        x_merged = self.out_proj(x_merged)
        x_merged = x_merged.permute(0, 2, 3, 1)
        x_merged = self.norm(x_merged)
        x_merged = x_merged.permute(0, 3, 1, 2)

        # Residual
        return residual + self.residual_scale.to(dtype) * x_merged


class VMamba3BlockVSSM(nn.Module):
    """VMamba-3 block for YOLO — drop-in replacement for VMamba2DBlock."""

    def __init__(
        self,
        dim,
        d_state=16,
        expand=2,
        headdim=64,
        drop_path=0.1,
        two_way_scan=False,
        use_rope=True,
        use_trapezoidal=True,
        mimo_rank=1,
        bc_bias=True,
        qk_norm=True,
    ):
        super().__init__()
        self.ss2d = SS2D_Mamba3(
            dim=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            two_way_scan=two_way_scan,
            use_rope=use_rope,
            use_trapezoidal=use_trapezoidal,
            mimo_rank=mimo_rank,
            bc_bias=bc_bias,
            qk_norm=qk_norm,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.ss2d(x))


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
