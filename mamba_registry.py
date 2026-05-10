# mamba_registry.py
import sys
import os

# Ensure workspace root is on path so mamba_decoder.py can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./libs/VMamba')

import torch.nn as nn
from ultralytics.nn.tasks import parse_model
import ultralytics.nn.modules as modules

from vmamba import VSSBlock, VSSBlock_Mamba2, VSSBlock_Mamba3


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
        dtype = x.dtype
        x = self.block(x.to('cuda'))
        return x.to(device=device, dtype=dtype)
    

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
        dtype = x.dtype
        x = self.block(x.to('cuda'))
        return x.to(device=device, dtype=dtype)


class VSSMBlock_Mamba3(nn.Module):
    """Wrapper around VMamba's VSSBlock_Mamba3 for YOLO integration (channel-first).
    YAML args: [dim, d_state, ssm_ratio, d_conv, headdim, chunk_size, ngroups]
      dim        : input/output channel count (set by YOLO from previous layer)
      d_state    : SSM state size (default 128, Mamba3 recommended)
      ssm_ratio  : expansion ratio for d_inner (default 2.0)
      d_conv     : depthwise conv kernel size (default 3)
      headdim    : head dimension — d_inner must be divisible (default 64)
      chunk_size : chunk length for parallel scan (default 64)
      ngroups    : number of B/C groups shared across heads (default 1)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        if len(args) >= 1:
            dim        = args[0]
            d_state    = args[1] if len(args) > 1 else 128
            ssm_ratio  = args[2] if len(args) > 2 else 2.0
            d_conv     = args[3] if len(args) > 3 else 3
            headdim    = args[4] if len(args) > 4 else 64
            chunk_size = args[5] if len(args) > 5 else 64
            ngroups    = args[6] if len(args) > 6 else 1
        else:
            raise Exception("VSSMBlock_Mamba3 requires at least dim argument")

        # Validate headdim divides d_inner
        d_inner = int(ssm_ratio * dim)
        assert d_inner % headdim == 0, (
            f"VSSMBlock_Mamba3: d_inner ({d_inner}) must be divisible by headdim ({headdim}). "
            f"Try headdim={d_inner // max(1, d_inner // headdim)} or adjust ssm_ratio."
        )

        self.block = VSSBlock_Mamba3(
            hidden_dim=dim,
            channel_first=True,      # YOLO uses (B, C, H, W)
            ssm_d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=d_conv,
            forward_type="m3",       # Mamba3 SISO kernel
            ssm_headdim=headdim,
            ssm_ngroups=ngroups,
            ssm_chunk_size=chunk_size,
            ssm_rmsnorm=True,
        ).to('cuda')

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        x = self.block(x.to('cuda'))
        return x.to(device=device, dtype=dtype)


class VSSMBlock_Mamba3_MIMO(nn.Module):
    """VSSMBlock_Mamba3 with MIMO kernel enabled.
    YAML args: [dim, d_state, ssm_ratio, d_conv, headdim, chunk_size, ngroups, mimo_rank]
      dim        : input/output channel count
      d_state    : SSM state size (default 128)
      ssm_ratio  : expansion ratio (default 2.0)
      d_conv     : depthwise conv kernel (default 3)
      headdim    : head dimension (default 64)
      chunk_size : chunk length — recommend <= 64//(mimo_rank*2), min 16 for TileLang warp partition (default 16)
      ngroups    : B/C groups (default 1)
      mimo_rank  : MIMO rank R (default 2)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        dim        = args[0]
        d_state    = args[1] if len(args) > 1 else 128
        ssm_ratio  = args[2] if len(args) > 2 else 2.0
        d_conv     = args[3] if len(args) > 3 else 3
        headdim    = args[4] if len(args) > 4 else 64
        chunk_size = args[5] if len(args) > 5 else 16
        ngroups    = args[6] if len(args) > 6 else 1
        mimo_rank  = args[7] if len(args) > 7 else 2

        d_inner = int(ssm_ratio * dim)
        assert d_inner % headdim == 0, (
            f"VSSMBlock_Mamba3_MIMO: d_inner ({d_inner}) must be divisible by headdim ({headdim})."
        )

        self.block = VSSBlock_Mamba3(
            hidden_dim=dim,
            channel_first=True,
            ssm_d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_conv=d_conv,
            forward_type="m3",
            ssm_headdim=headdim,
            ssm_ngroups=ngroups,
            ssm_chunk_size=chunk_size,
            ssm_rmsnorm=True,
            is_mimo=True,
            mimo_rank=mimo_rank,
        ).to('cuda')

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        x = self.block(x.to('cuda'))
        return x.to(device=device, dtype=dtype)




# Register custom modules with ultralytics
_original_parse_model = parse_model

def mamba_parse_model(d, ch, verbose=True):
    """Custom parse_model that registers custom Mamba modules"""
    import sys
    
    original_globals = sys.modules['ultralytics.nn.tasks'].__dict__
    original_globals['VSSMBlock'] = VSSMBlock
    original_globals['VSSMBlock_Mamba2'] = VSSMBlock_Mamba2
    original_globals['VSSMBlock_Mamba3'] = VSSMBlock_Mamba3
    original_globals['VSSMBlock_Mamba3_MIMO'] = VSSMBlock_Mamba3_MIMO
    
    return _original_parse_model(d, ch, verbose)

import ultralytics.nn.tasks as tasks
tasks.parse_model = mamba_parse_model
modules.parse_model = mamba_parse_model
modules.VSSMBlock = VSSMBlock
modules.VSSMBlock_Mamba2 = VSSMBlock_Mamba2
modules.VSSMBlock_Mamba3 = VSSMBlock_Mamba3
modules.VSSMBlock_Mamba3_MIMO = VSSMBlock_Mamba3_MIMO

