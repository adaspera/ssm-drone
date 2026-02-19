import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossScan(nn.Module):
    """Cross-Scan Module for VMamba - converts 2D to 4 directional sequences"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create 4 directional scans
        scans = []
        
        # Scan 1: Left to right, top to bottom (standard)
        scan1 = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        scans.append(scan1)
        
        # Scan 2: Right to left, top to bottom
        x_flip_lr = torch.flip(x, dims=[3])
        scan2 = x_flip_lr.flatten(2).transpose(1, 2)
        scans.append(scan2)
        
        # Scan 3: Top to bottom, left to right (transpose)
        x_permute = x.permute(0, 1, 3, 2)  # [B, C, W, H]
        scan3 = x_permute.flatten(2).transpose(1, 2)
        scans.append(scan3)
        
        # Scan 4: Bottom to top, left to right
        x_flip_ud = torch.flip(x_permute, dims=[3])
        scan4 = x_flip_ud.flatten(2).transpose(1, 2)
        scans.append(scan4)
        
        return scans, (H, W)
    
    def merge(self, scans, original_size, inner_dim):
        """Merge 4 directional scans back to 2D (element-wise summation)"""
        B, C, H, W = original_size
        device = scans[0].device
        dtype = scans[0].dtype
        
        # Initialize output tensor with inner_dim channels
        merged = torch.zeros(B, inner_dim, H, W, device=device, dtype=dtype)
        
        # Scan 1: standard
        scan1 = scans[0].transpose(1, 2).reshape(B, inner_dim, H, W)
        merged += scan1
        
        # Scan 2: flip back
        scan2 = scans[1].transpose(1, 2).reshape(B, inner_dim, H, W)
        merged += torch.flip(scan2, dims=[3])
        
        # Scan 3: permute back
        scan3 = scans[2].transpose(1, 2).reshape(B, inner_dim, W, H)
        merged += scan3.permute(0, 1, 3, 2)
        
        # Scan 4: flip and permute back
        scan4 = scans[3].transpose(1, 2).reshape(B, inner_dim, W, H)
        merged += torch.flip(scan4, dims=[3]).permute(0, 1, 3, 2)
        
        return merged


class CrossScan2(nn.Module):
    """Two-Scan Module for VMamba - converts 2D to 2 directional sequences (faster)"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        scans = []
        
        # Scan 1: Left to right, top to bottom (standard row-major)
        scan1 = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        scans.append(scan1)
        
        # Scan 2: Right to left, bottom to top (alternating direction)
        x_flip_lr = torch.flip(x, dims=[3])  # Flip left-right
        x_flip_ud = torch.flip(x_flip_lr, dims=[2])  # Flip up-down
        scan2 = x_flip_ud.flatten(2).transpose(1, 2)
        scans.append(scan2)
        
        return scans, (H, W)
    
    def merge(self, scans, original_size, inner_dim):
        """Merge 2 directional scans back to 2D (element-wise summation)"""
        B, C, H, W = original_size
        device = scans[0].device
        dtype = scans[0].dtype
        
        # Initialize output tensor
        merged = torch.zeros(B, inner_dim, H, W, device=device, dtype=dtype)
        
        # Scan 1: standard row-major
        scan1 = scans[0].transpose(1, 2).reshape(B, inner_dim, H, W)
        merged += scan1
        
        # Scan 2: reverse both dimensions
        scan2 = scans[1].transpose(1, 2).reshape(B, inner_dim, H, W)
        # Flip back both dimensions to original orientation
        merged += torch.flip(scan2, dims=[2, 3])
        
        return merged

class SS2D(nn.Module):
    """
    2D Selective Scan Module for VMamba
    """
    
    def __init__(self, dim, d_state=16, expand=2, d_conv=3, two_way_scan=False):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.d_inner = int(dim * expand)
        
        # Cross-scan module
        self.cross_scan = CrossScan(dim) if two_way_scan == False else CrossScan2(dim)
        
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
        from mamba_ssm import Mamba
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
        
        # ----- Step 1: Cross-scan -----
        scans, _ = self.cross_scan(x)
        
        # Process each scan
        processed_scans = []
        for scan in scans:
            original_len = scan.shape[1]
            
            # Input projection (expand channels)
            proj = self.in_proj(scan.to(dtype))
            x_ssm, x_gate = proj.chunk(2, dim=-1)
            
            # Path 1: 1D conv
            x_ssm = x_ssm.transpose(1, 2)
            x_ssm = self.conv1d(x_ssm)
            x_ssm = x_ssm.transpose(1, 2)
            x_ssm = F.silu(x_ssm)
            
            # Mamba processing
            if device.type != 'cpu':
                x_ssm = self.mamba(x_ssm)
            
            # Path 2: Gating
            x_gate = F.silu(x_gate)
            
            # Ensure correct length
            if x_ssm.shape[1] != original_len:
                x_ssm = x_ssm[:, :original_len, :]
            if x_gate.shape[1] != original_len:
                x_gate = x_gate[:, :original_len, :]
            
            # Combine
            scan_out = x_ssm * x_gate
            processed_scans.append(scan_out)
        
        # ----- Step 2: Merge scans -----
        x_out = self.cross_scan.merge(processed_scans, (B, C, H, W), self.d_inner)
        
        # ----- Step 3: Output projection -----
        x_out = self.out_proj(x_out)
        
        # ----- Step 4: Normalization -----
        x_out = x_out.permute(0, 2, 3, 1)
        x_out = self.norm(x_out)
        x_out = x_out.permute(0, 3, 1, 2)
        
        # Residual
        x_out = residual + self.residual_scale.to(dtype) * x_out
        
        return x_out


class VMamba2DBlock(nn.Module):
    """
    VMamba block for YOLO
    """
    
    def __init__(self, dim, d_state=16, expand=2, d_conv=3, drop_path=0.1, two_way_scan=False):
        super().__init__()
        self.dim = dim
        
        self.ss2d = SS2D(
            dim=dim,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            two_way_scan=two_way_scan
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