import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('./libs/VMamba')
from vmamba import cross_scan_fn, cross_merge_fn


# Using optimized cross_scan_fn and cross_merge_fn from VMamba library


class SS2D(nn.Module):
    """
    2D Selective Scan Module for VMamba
    """
    
    def __init__(self, dim, d_state=16, expand=2, d_conv=3, 
                 two_way_scan=False, mamba_type='v2'):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.d_inner = int(dim * expand)
        self.two_way_scan = two_way_scan
        self.scans = 2 if two_way_scan else 0  # 0=4-way cross, 2=bidirectional
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=1,
            groups=self.d_inner,
            bias=False
        )
        
        # Mamba module
        from mamba_ssm import Mamba, Mamba2
        if mamba_type == 'v2':
            self.mamba = Mamba2(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=self.d_conv,
                expand=1,
            )
        else:
            self.mamba = Mamba(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=self.d_conv,
                expand=1,
            )
        
        # Output projection (applied AFTER merge)
        self.out_proj = nn.Conv2d(self.d_inner, dim, 1, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
        # Stabilization
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        device = x.device
        dtype = x.dtype
        
        # ----- Step 1: Cross-scan (library optimized) -----
        # Output: (B, K, C, L) where K=4 for cross-scan or K=2 for bidirectional
        x_scanned = cross_scan_fn(
            x, 
            in_channel_first=True, 
            out_channel_first=True, 
            scans=self.scans,
            force_torch=False
        )
        
        B, K, C, L = x_scanned.shape  # K=4 or 2, L=H*W
        
        # Reshape for processing: (B, K, C, L) -> (B*K, L, C)
        x_scanned = x_scanned.view(B * K, L, C)
        
        # Process each scan
        # Input projection (expand channels)
        proj = self.in_proj(x_scanned.to(dtype))
        x_ssm, x_gate = proj.chunk(2, dim=-1)
        
        # Path 1: 1D conv
        x_ssm = x_ssm.transpose(1, 2)  # (B*K, d_inner, L)
        x_ssm = self.conv1d(x_ssm)
        x_ssm = x_ssm.transpose(1, 2)  # (B*K, L, d_inner)
        x_ssm = F.silu(x_ssm)
        
        # Mamba processing
        if device.type != 'cpu':
            x_ssm = self.mamba(x_ssm)
        
        # Path 2: Gating
        x_gate = F.silu(x_gate)
        
        # Combine paths
        x_out = x_ssm * x_gate  # (B*K, L, d_inner)
        
        # Reshape back for merge: (B*K, L, d_inner) -> (B, K, d_inner, L)
        x_out = x_out.transpose(1, 2).view(B, K, self.d_inner, L)
        
        # Reshape to 5D for cross_merge_fn: (B, K, d_inner, L) -> (B, K, d_inner, H, W)
        x_out = x_out.reshape(B, K, self.d_inner, H, W)
        
        # ----- Step 2: Cross-merge (library optimized) -----
        # Output: (B, d_inner, L) - flattened
        x_merged = cross_merge_fn(
            x_out,
            in_channel_first=True,
            out_channel_first=True,
            scans=self.scans,
            force_torch=False
        )
        
        # Reshape back to spatial: (B, d_inner, L) -> (B, d_inner, H, W)
        x_merged = x_merged.reshape(B, self.d_inner, H, W)
        
        # ----- Step 3: Output projection -----
        x_merged = self.out_proj(x_merged)
        
        # ----- Step 4: Normalization -----
        x_merged = x_merged.permute(0, 2, 3, 1)
        x_merged = self.norm(x_merged)
        x_merged = x_merged.permute(0, 3, 1, 2)
        
        # Residual
        x_out = residual + self.residual_scale.to(dtype) * x_merged
        
        return x_out


class VMamba2DBlock(nn.Module):
    """
    VMamba block for YOLO
    """
    
    def __init__(self, dim, d_state=16, expand=2, d_conv=3, drop_path=0.1, two_way_scan=False,
                 mamba_type='v2'):
        super().__init__()
        self.dim = dim
        
        self.ss2d = SS2D(
            dim=dim,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            two_way_scan=two_way_scan,
            mamba_type=mamba_type
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        return x + self.drop_path(self.ss2d(x))



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor