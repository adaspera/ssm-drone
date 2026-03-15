# mamba_block.py
import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel
from mamba_ssm import Mamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class Mamba2DStable(nn.Module):
    """Stabilized Mamba block for 2D inputs"""
    
    def __init__(self, dim, d_state, expand, d_conv):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv

        self._device_str = "cuda"
        
        
        # Store parameters
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv

        self._device_str = "cuda"
        
        print(f"Mamba2D: Initializing (dim={dim})")
        
        # Initialize Mamba with lower default params for stability
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,  # Cap d_state
            d_conv=d_conv,
            expand=expand,
        )
        
        # Layer norm with epsilon
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        
        # Additional stabilization
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # Learnable scale
        self.dropout = nn.Dropout(0.1)

        self.to(self._device_str)
        
        self._nan_count = 0
        self.chunk_size = 64  # Smaller chunks for stability
        self._cpu_passes = 0
    
    def _stabilize(self, x):
        """Multiple stabilization techniques"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            self._nan_count += 1
            if self._nan_count < 10:
                print(f"⚠️ Mamba2D: Found NaN/Inf (total: {self._nan_count})")
            
            # Replace NaNs with zeros
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip extreme values
        x = torch.clamp(x, -10.0, 10.0)
        
        return x
    
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # Handle CPU initialization
        if device.type == 'cpu' and self._cpu_passes < 2:
            self._cpu_passes += 1
            return torch.randn(B, C, H, W, device=device) * 0.01
        
        # Initial stabilization
        x = self._stabilize(x)
        
        # Reshape and normalize
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Apply layer norm
        x_seq = self.norm(x_seq)
        x_seq = self._stabilize(x_seq)
        

        x_seq = self.mamba(x_seq)
        x_seq = self._stabilize(x_seq)

        # Final normalization
        x_seq = F.layer_norm(x_seq, [x_seq.size(-1)])
        x_seq = self.dropout(x_seq)
        
        # Reshape back
        x_out = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        return self._stabilize(x_out)
