import torch
import torch.nn as nn
import vmamba
from vmamba import VSSM
from copy import deepcopy as torch_deepcopy

class VMambaBackbone(nn.Module):
    """VMamba-based backbone using timm models with integrated feature reduction"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        import timm
        
        # Create VMamba model from timm
        self.model = timm.create_model(
            'vanilla_vmamba_tiny',
            pretrained=False,
            features_only=False,
        )
        
        # Remove classifier head
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity()
        
        # Disable inplace operations in VMamba to avoid gradient issues
        self._disable_inplace_ops(self.model)
        
        # Channel reduction layers for YOLO
        # VMamba outputs channels [192, 384, 768, 768] at different scales in BHWC format
        # Use layers 0, 1, 2 for P2, P3, P4 (stride 8, 16, 32)
        # Reduce to smaller sizes [64, 128, 256]
        self.reduce_p2 = nn.Conv2d(192, 64, 1)   # layer_0: 192ch -> 64ch
        self.reduce_p3 = nn.Conv2d(384, 128, 1)  # layer_1: 384ch -> 128ch
        self.reduce_p4 = nn.Conv2d(768, 256, 1)  # layer_2: 768ch -> 256ch
        
        # Feature hook tracking
        self.features_dict = {}
        self._register_hooks()
    
    def _disable_inplace_ops(self, module):
        """Recursively disable inplace operations in the model"""
        for child in module.children():
            if isinstance(child, nn.SiLU):
                child.inplace = False
            elif isinstance(child, nn.ReLU):
                child.inplace = False
            elif isinstance(child, nn.GELU):
                child.inplace = False
            # Recursively process child modules
            self._disable_inplace_ops(child)
    
    def __deepcopy__(self, memo):
        """Custom deepcopy to handle timm models that can't be deepcopied"""
        # Create a new instance without deepcopying the timm model or Conv2d layers
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k == 'model':
                # Create a fresh model instead of deepcopying
                import timm
                new_model = timm.create_model(
                    'vanilla_vmamba_tiny',
                    pretrained=False,
                    features_only=False,
                )
                if hasattr(new_model, 'head'):
                    new_model.head = nn.Identity()
                setattr(result, k, new_model)
            elif k in ['reduce_p2', 'reduce_p3', 'reduce_p4']:
                # Recreate Conv2d layers instead of deepcopying
                if k == 'reduce_p2':
                    new_layer = nn.Conv2d(192, 64, 1)
                elif k == 'reduce_p3':
                    new_layer = nn.Conv2d(384, 128, 1)
                else:  # reduce_p4
                    new_layer = nn.Conv2d(768, 256, 1)
                setattr(result, k, new_layer)
            elif k == 'features_dict':
                # Reset features_dict to empty
                setattr(result, k, {})
            else:
                # Standard deepcopy for other attributes
                try:
                    setattr(result, k, torch_deepcopy(v, memo))
                except:
                    # If deepcopy fails, just reference the original
                    setattr(result, k, v)
        
        return result
    
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
        # Store original device (mamba only works on CUDA, so force all computation there)
        original_device = x.device
        original_dtype = x.dtype
        
        # Disable inplace ops right before forward to ensure they're disabled
        self._disable_inplace_ops(self.model)
        
        # Move input to CUDA for mamba computation
        x_cuda = x.to('cuda')
        
        # Move VMamba model to CUDA (always stays there)
        self.model = self.model.to('cuda')
        self.reduce_p2 = self.reduce_p2.to('cuda')
        self.reduce_p3 = self.reduce_p3.to('cuda')
        self.reduce_p4 = self.reduce_p4.to('cuda')
        
        # Disable inplace ops again after moving to device
        self._disable_inplace_ops(self.model)
        
        self.features_dict = {}
        with torch.set_grad_enabled(True):
            _ = self.model(x_cuda)
        
        # Extract the three features we need and reduce them
        features = []
        
        # P2: stride-8 (192 channels -> 64) from layer_0
        if 'layer_0' in self.features_dict:
            f_p2 = self.features_dict['layer_0'].clone()
            # VMamba outputs BHWC format, convert to BCHW for Conv2d
            f_p2 = f_p2.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            f_p2 = self.reduce_p2(f_p2)
            # Move back to original device
            features.append(f_p2.to(original_device))
        
        # P3: stride-16 (384 channels -> 128) from layer_1
        if 'layer_1' in self.features_dict:
            f_p3 = self.features_dict['layer_1'].clone()
            # VMamba outputs BHWC format, convert to BCHW for Conv2d
            f_p3 = f_p3.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            f_p3 = self.reduce_p3(f_p3)
            # Move back to original device
            features.append(f_p3.to(original_device))
        
        # P4: stride-32 (768 channels -> 256) from layer_2
        if 'layer_2' in self.features_dict:
            f_p4 = self.features_dict['layer_2'].clone()
            # VMamba outputs BHWC format, convert to BCHW for Conv2d
            f_p4 = f_p4.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            f_p4 = self.reduce_p4(f_p4)
            # Move back to original device
            features.append(f_p4.to(original_device))
        
        if not features:
            print("WARNING: No features captured!")
            features = [x]
        
        # Return as a list with proper shapes for direct YOLO head input
        return features