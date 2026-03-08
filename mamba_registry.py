# mamba_registry.py
import torch
import torch.nn as nn
from ultralytics.nn.tasks import parse_model
import ultralytics.nn.modules as modules

import sys
sys.path.append('./libs/VMamba')  # Add VMamba to path

import vmamba
from vmamba import VSSBlock, VSSBlock_Mamba2

from mamba_block import Mamba2DStable
from vmamba_block import VMamba2DBlock
from mamba3_block import VMamba3BlockVSSM
from mamba_ssm import Mamba2

class MambaBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 4:
            print(args)
            input_chanels, output_channels, d_state, expand = args[0], args[1], args[2], args[3]
            d_conv = args[4] if len(args) > 4 else 4 
        else: 
            raise Exception("Wrong arg count")

        if not hasattr(self, 'mamba'):
            self.mamba = Mamba2DStable(
                dim=input_chanels,  # Use actual input channels
                d_state=d_state,
                expand=expand,
                d_conv=d_conv
            )
            
            # Only add projection if channels need to change
            if input_chanels != output_channels:
                self.proj = nn.Conv2d(input_chanels, output_channels, 1)
            else:
                self.proj = None
    
    def forward(self, x):
        self.mamba.to(x.device)
        x_out = self.mamba(x)
        
        if self.proj:
            self.proj.to(x.device)
            x_out = self.proj(x_out)
        
        return x_out

class VMambaBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 4:
            print(args)
            input_chanels, d_state, expand, d_conv = args[0], args[1], args[2], args[3]
            if len(args) > 4:
                mamba_type = args[4]
            else:
                mamba_type = 'v1'
        else: 
            raise Exception("Wrong arg count")

        if not hasattr(self, 'mamba'):
            self.mamba = VMamba2DBlock(
                dim=input_chanels,  # Use actual input channels
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                mamba_type=mamba_type
            )

    
    def forward(self, x):
        self.mamba.to(x.device)
        x_out = self.mamba(x)
        
        return x_out

class VSSMBlock(nn.Module):
    """Wrapper around VMamba's VSSBlock for YOLO integration (channel-first).
    YAML args: [dim, d_state, ssm_ratio, d_conv]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 1:
            print(args)
            dim = args[0]
            d_state  = args[1] if len(args) > 1 else 16
            ssm_ratio = args[2] if len(args) > 2 else 2.0
            d_conv   = args[3] if len(args) > 3 else 3
        else:
            raise Exception("VSSMBlock requires at least dim argument")

        self.block = VSSBlock(
            hidden_dim=dim,
            channel_first=True,   # YOLO uses (B, C, H, W)
            ssm_d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=d_conv,
            forward_type="v2",
        ).to('cuda')

    def forward(self, x):
        device = x.device
        x = self.block(x.to('cuda'))
        return x.to(device)  # Ensure output is on same device as input
    

class VSSMBlock_Mamba2(nn.Module):
    """Wrapper around VMamba's VSSBlock_Mamba2 for YOLO integration (channel-first).
    YAML args: [dim, d_state, ssm_ratio, d_conv, headdim, chunk_size, ngroups]
      dim        : input/output channel count (set by YOLO from previous layer)
      d_state    : SSM state size (default 64, Mamba2 recommended)
      ssm_ratio  : expansion ratio for d_inner (default 2.0)
      d_conv     : depthwise conv kernel size (default 3)
      headdim    : head dimension inside SSD — d_inner must be divisible (default 64)
      chunk_size : SSD chunk length for parallel scan (default 256)
      ngroups    : number of B/C groups shared across heads (default 1)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 1:
            dim        = args[0]
            d_state    = args[1] if len(args) > 1 else 64
            ssm_ratio  = args[2] if len(args) > 2 else 2.0
            d_conv     = args[3] if len(args) > 3 else 3
            headdim    = args[4] if len(args) > 4 else 64
            chunk_size = args[5] if len(args) > 5 else 256
            ngroups    = args[6] if len(args) > 6 else 1
        else:
            raise Exception("VSSMBlock_Mamba2 requires at least dim argument")

        # Validate headdim divides d_inner
        d_inner = int(ssm_ratio * dim)
        assert d_inner % headdim == 0, (
            f"VSSMBlock_Mamba2: d_inner ({d_inner}) must be divisible by headdim ({headdim}). "
            f"Try headdim={d_inner // max(1, d_inner // headdim)} or adjust ssm_ratio."
        )

        self.block = VSSBlock_Mamba2(
            hidden_dim=dim,
            channel_first=True,      # YOLO uses (B, C, H, W)
            ssm_d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=d_conv,
            forward_type="ssd",      # Mamba2 SSD kernel
            ssm_headdim=headdim,
            ssm_ngroups=ngroups,
            ssm_chunk_size=chunk_size,
            ssm_rmsnorm=True,
        ).to('cuda')

    def forward(self, x):
        device = x.device
        x = self.block(x.to('cuda'))
        return x.to(device)


class VMambaBlock2Way(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 4:
            print(args)
            input_chanels, d_state, expand, d_conv = args[0], args[1], args[2], args[3]
        else: 
            raise Exception("Wrong arg count")

        if not hasattr(self, 'mamba'):
            self.mamba = VMamba2DBlock(
                dim=input_chanels,  # Use actual input channels
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                two_way_scan=True
            )

    
    def forward(self, x):
        self.mamba.to(x.device)
        x_out = self.mamba(x)
        
        return x_out


class VMamba3Block(nn.Module):
    """YOLO wrapper for VMamba-3 block (cross-scan + Mamba-3 SSM).
    YAML args: [dim, d_state, expand, headdim, mimo_rank]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) >= 2:
            print(args)
            input_channels = args[0]
            d_state = args[1] if len(args) > 1 else 16
            expand = args[2] if len(args) > 2 else 2
            headdim = args[3] if len(args) > 3 else 64
            mimo_rank = args[4] if len(args) > 4 else 1
        else:
            raise Exception("VMamba3Block requires at least [dim, d_state]")

        self.mamba = VMamba3BlockVSSM(
            dim=input_channels,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            mimo_rank=mimo_rank,
        )

    def forward(self, x):
        self.mamba.to(x.device)
        return self.mamba(x)




# Register custom modules with ultralytics
_original_parse_model = parse_model

def mamba_parse_model(d, ch, verbose=True):
    """Custom parse_model that registers custom Mamba modules"""
    import sys
    
    original_globals = sys.modules['ultralytics.nn.tasks'].__dict__
    original_globals['MambaBlock'] = MambaBlock
    original_globals['VMambaBlock'] = VMambaBlock
    original_globals['VMambaBlock2Way'] = VMambaBlock2Way
    original_globals['VSSMBlock'] = VSSMBlock
    original_globals['VMamba3Block'] = VMamba3Block
    original_globals['VSSMBlock_Mamba2'] = VSSMBlock_Mamba2
    
    try:
        return _original_parse_model(d, ch, verbose)
    finally:
        if 'MambaBlock' in original_globals:
            del original_globals['MambaBlock']

import ultralytics.nn.tasks as tasks
tasks.parse_model = mamba_parse_model
modules.parse_model = mamba_parse_model
modules.MambaBlock = MambaBlock
modules.VMambaBlock = VMambaBlock
modules.VMambaBlock2Way = VMambaBlock2Way
modules.VSSMBlock = VSSMBlock
modules.VMamba3Block = VMamba3Block
modules.VSSMBlock_Mamba2 = VSSMBlock_Mamba2
