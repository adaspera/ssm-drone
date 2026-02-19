# mamba_registry.py
import torch
import torch.nn as nn
from ultralytics.nn.tasks import parse_model
import ultralytics.nn.modules as modules

import sys
sys.path.append('./libs/VMamba')  # Add VMamba to path

import vmamba
from vmamba import VSSM

from mamba_block import Mamba2DStable
from vmamba_block import VMamba2DBlock

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
        else: 
            raise Exception("Wrong arg count")

        if not hasattr(self, 'mamba'):
            self.mamba = VMamba2DBlock(
                dim=input_chanels,  # Use actual input channels
                d_state=d_state,
                expand=expand,
                d_conv=d_conv
            )

    
    def forward(self, x):
        self.mamba.to(x.device)
        x_out = self.mamba(x)
        
        return x_out

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



class VMambaBackbone(nn.Module):
    """VMamba-based backbone using timm models with integrated feature reduction"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        print("\n" + "="*50)
        print("VMambaBackbone Initialization")
        print("="*50)
        
        import timm
        
        # Create VMamba model from timm
        print("Creating vanilla_vmamba_tiny model from timm...")
        self.model = timm.create_model(
            'vanilla_vmamba_tiny',
            pretrained=False,
            features_only=False,
        ).to('cuda')
        print("Model created successfully")
        
        # Remove classifier head
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity().to('cuda')
        
        # Channel reduction layers for YOLO
        # VMamba outputs channels approximately [96, 192, 384, 768] at different scales
        # Reduce to standard YOLO sizes [128, 256, 512]
        self.reduce_p2 = nn.Conv2d(192, 128, 1).to('cuda')  # P2 stride-8
        self.reduce_p3 = nn.Conv2d(384, 256, 1).to('cuda')  # P3 stride-16
        self.reduce_p4 = nn.Conv2d(768, 512, 1).to('cuda')  # P4 stride-32
        
        # Feature hook tracking
        self.features_dict = {}
        self._register_hooks()

        print("Hooks registered and reduction layers created")
        print("="*50 + "\n")
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        def get_hook(name):
            def hook(module, input, output):
                self.features_dict[name] = output
            return hook
        
        if hasattr(self.model, 'layers'):
            # Register hooks for each layer
            for i, layer in enumerate(self.model.layers):
                layer.register_forward_hook(get_hook(f'layer_{i}'))
        
    def forward(self, x):
        """Forward pass and return reduced features for YOLO detection"""
        # Store original device and move to CUDA for VMamba computation
        original_device = x.device
        x = x.to('cuda')
        self.model.to('cuda')
        self.reduce_p2.to('cuda')
        self.reduce_p3.to('cuda')
        self.reduce_p4.to('cuda')
        
        self.features_dict = {}
        _ = self.model(x)
        
        # Extract the three features we need and reduce them
        features = []
        
        # P2: stride-8 (192 channels -> 128)
        if 'layer_1' in self.features_dict:
            f_p2 = self.features_dict['layer_1']
            f_p2 = self.reduce_p2(f_p2)
            features.append(f_p2.to(original_device))
        
        # P3: stride-16 (384 channels -> 256)
        if 'layer_2' in self.features_dict:
            f_p3 = self.features_dict['layer_2']
            f_p3 = self.reduce_p3(f_p3)
            features.append(f_p3.to(original_device))
        
        # P4: stride-32 (768 channels -> 512)
        if 'layer_3' in self.features_dict:
            f_p4 = self.features_dict['layer_3']
            f_p4 = self.reduce_p4(f_p4)
            features.append(f_p4.to(original_device))
        
        if not features:
            print("WARNING: No features captured!")
            features = [x.to(original_device)]
        
        # Return as a list with proper shapes for direct YOLO head input
        return features


class VMambaDetect(nn.Module):
    """Custom Detect layer that properly handles VMamba backbone list output"""
    def __init__(self, nc=80, anchors=3, ch=(128, 256, 512), *args, **kwargs):
        super().__init__()
        from ultralytics.nn.modules import Detect
        # Delegate to standard Detect (nc and ch only)
        self.detect_module = Detect(nc=nc, ch=ch)
        self.nc = nc
        self.stride = None
    
    def forward(self, x):
        """Accept list/tuple or single tensor input"""
        # VMambaBackbone returns a list [P2, P3, P4]
        # Detect expects this structure, so just pass through
        if isinstance(x, (list, tuple)):
            return self.detect_module(x)
        return self.detect_module([x])


# Register custom modules with ultralytics
_original_parse_model = parse_model

def mamba_parse_model(d, ch, verbose=True):
    """Custom parse_model that registers custom Mamba modules"""
    import sys
    
    original_globals = sys.modules['ultralytics.nn.tasks'].__dict__
    original_globals['MambaBlock'] = MambaBlock
    original_globals['VMambaBlock'] = VMambaBlock
    original_globals['VMambaBlock2Way'] = VMambaBlock2Way
    original_globals['VMambaBackbone'] = VMambaBackbone
    original_globals['VMambaDetect'] = VMambaDetect
    
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
modules.VMambaBackbone = VMambaBackbone
modules.VMambaDetect = VMambaDetect
