"""
Modular Vision Backbone Module
Provides plug-in vision encoders with frozen weights for feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Callable


def get_backbone(name: str, device: torch.device) -> Tuple[nn.Module, int, Callable]:
    """
    Get a frozen vision backbone with preprocessing function.
    
    Args:
        name: Backbone identifier (mobilenet_v2, resnet18, resnet34, resnet50, clip_vit_b32,
              dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
        device: Device to load model on
        
    Returns:
        (model, feat_dim, preprocess_fn)
        - model: Frozen backbone in eval mode
        - feat_dim: Output feature dimension
        - preprocess_fn: Function that takes numpy RGB uint8 (H,W,3) -> torch tensor (1,3,H,W)
    """
    
    name = name.lower()
    
    # ImageNet normalization (used by torchvision models and DINOv2)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    def torchvision_preprocess(img_np: np.ndarray) -> torch.Tensor:
        """Preprocess numpy RGB image for torchvision models."""
        # img_np is (H, W, 3) uint8 RGB
        img_tensor = torch.from_numpy(img_np).float() / 255.0  # (H, W, 3) in [0, 1]
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        img_tensor = transforms.Resize((224, 224), antialias=True)(img_tensor)
        img_tensor = imagenet_normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
        return img_tensor.to(device)
    
    # MobileNetV2
    if name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        # Remove classifier, keep features only
        model = model.features
        feat_dim = 1280  # MobileNetV2 output channels
        preprocess_fn = torchvision_preprocess
        
    # ResNet18
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Remove final fc layer, use pooled features
        model = nn.Sequential(*list(model.children())[:-1])  # Remove fc
        feat_dim = 512
        preprocess_fn = torchvision_preprocess
        
    # ResNet34
    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        feat_dim = 512
        preprocess_fn = torchvision_preprocess
        
    # ResNet50
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        feat_dim = 2048
        preprocess_fn = torchvision_preprocess
        
    # CLIP ViT-B/32
    elif name == 'clip_vit_b32':
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "CLIP backbone requires open_clip. Install with:\n"
                "  pip install open-clip-torch\n"
                "Or use a different backbone (mobilenet_v2, resnet18, etc.)"
            )
        
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.visual  # Vision encoder only
        feat_dim = 512  # CLIP ViT-B/32 output dim
        
        def clip_preprocess(img_np: np.ndarray) -> torch.Tensor:
            """Preprocess for CLIP."""
            # Convert numpy to PIL-like tensor
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            # Apply CLIP preprocessing (resize, normalize)
            img_tensor = transforms.Resize((224, 224), antialias=True)(img_tensor)
            img_tensor = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )(img_tensor)
            return img_tensor.to(device)
        
        preprocess_fn = clip_preprocess
        
    # DINOv2 models
    elif name in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']:
        try:
            # NOTE: DINOv2 official repo requires Python 3.10+ (uses | union syntax)
            # Workaround for Python 3.8: Load weights manually
            
            # DINOv2 feature dimensions
            feat_dims = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536,
            }
            feat_dim = feat_dims[name]
            
            # Try torch.hub first (works if Python >= 3.10)
            try:
                model = torch.hub.load('facebookresearch/dinov2', name, trust_repo=True)
                print(f"[Vision] Loaded {name} via torch.hub (feat_dim={feat_dim})", flush=True)
            except (TypeError, SyntaxError) as e:
                # Python 3.8 compatibility: use timm's DINOv2 models (slightly different but compatible)
                print(f"[Vision] torch.hub failed (Python 3.8), using timm's DINOv2 implementation...", flush=True)
                
                # Import timm (required for ViT models)
                try:
                    import timm
                except ImportError:
                    raise ImportError(
                        "DINOv2 on Python 3.8 requires timm. Install with:\n"
                        "  pip install timm\n"
                    )
                
                # Map to timm's vit_*_patch14_reg4_dinov2 models (224x224 native)
                timm_model_names = {
                    'dinov2_vits14': 'vit_small_patch14_reg4_dinov2.lvd142m',
                    'dinov2_vitb14': 'vit_base_patch14_reg4_dinov2.lvd142m',
                    'dinov2_vitl14': 'vit_large_patch14_reg4_dinov2.lvd142m',
                    'dinov2_vitg14': 'vit_giant_patch14_reg4_dinov2.lvd142m',
                }
                
                timm_model_name = timm_model_names.get(name)
                if not timm_model_name:
                    raise ValueError(f"No timm equivalent for {name}")
                
                # Create model via timm (uses 224x224 by default)
                model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0,  # Remove classification head
                )
                
                print(f"[Vision] Loaded {name} via timm (feat_dim={feat_dim})", flush=True)
            
            def dinov2_preprocess(img_np: np.ndarray) -> torch.Tensor:
                """Preprocess numpy RGB image for DINOv2."""
                # DINOv2 (timm) models expect 518x518 by default
                # img_np is (H, W, 3) uint8 RGB
                img_tensor = torch.from_numpy(img_np).float() / 255.0  # (H, W, 3) in [0, 1]
                img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
                img_tensor = transforms.Resize((518, 518), antialias=True)(img_tensor)
                img_tensor = imagenet_normalize(img_tensor)
                img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 518, 518)
                return img_tensor.to(device)
            
            preprocess_fn = dinov2_preprocess
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {name}. Error: {e}\n"
                f"For Python 3.8, install timm: pip install timm\n"
                f"For Python 3.10+, torch.hub should work directly."
            )
        
    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Supported: mobilenet_v2, resnet18, resnet34, resnet50, clip_vit_b32, "
            f"dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14"
        )
    
    # Freeze all parameters
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model, feat_dim, preprocess_fn


class ProjectionMLP(nn.Module):
    """
    Trainable projection network to reduce vision features to lower dimension.
    """
    def __init__(self, in_dim: int, proj_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim),
            nn.LayerNorm(proj_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_dim) vision features
        Returns:
            (batch, proj_dim) projected features
        """
        return self.net(x)


def extract_vision_tokens(backbone: nn.Module, x: torch.Tensor, backbone_name: str) -> torch.Tensor:
    """
    Extract spatial tokens from vision backbone.
    
    Args:
        backbone: Vision model (frozen)
        x: (B, 3, H, W) preprocessed image
        backbone_name: Backbone identifier
    
    Returns:
        tokens: (B, N, C) where N is number of spatial tokens, C is token dimension
    """
    with torch.no_grad():
        # DINOv2/ViT models
        if 'dinov2' in backbone_name or 'vit' in backbone_name or 'clip' in backbone_name:
            # ViT models output tokens directly or can be accessed via forward_features
            if hasattr(backbone, 'forward_features'):
                # timm models
                tokens = backbone.forward_features(x)  # (B, N+1, C) with CLS token
                # Remove CLS token (first token)
                if tokens.shape[1] > 1:
                    tokens = tokens[:, 1:, :]  # (B, N, C)
                return tokens
            elif hasattr(backbone, 'get_intermediate_layers'):
                # Official DINOv2
                output = backbone.get_intermediate_layers(x, n=1)[0]  # (B, N+1, C)
                # Remove CLS token
                if output.shape[1] > 1:
                    tokens = output[:, 1:, :]
                else:
                    tokens = output
                return tokens
            else:
                # Fallback: run forward and try to extract
                output = backbone(x)
                if isinstance(output, torch.Tensor):
                    if output.dim() == 3:
                        # Already tokens (B, N, C)
                        return output
                    elif output.dim() == 2:
                        # Pooled features (B, C) -> treat as single token
                        return output.unsqueeze(1)  # (B, 1, C)
                    elif output.dim() == 4:
                        # Spatial features (B, C, H, W) -> flatten to tokens
                        B, C, H, W = output.shape
                        tokens = output.reshape(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
                        return tokens
        
        # CNN models (MobileNet, ResNet, etc.)
        else:
            output = backbone(x)
            if output.dim() == 4:
                # (B, C, H, W) -> (B, H*W, C)
                B, C, H, W = output.shape
                tokens = output.reshape(B, C, H*W).permute(0, 2, 1)
                return tokens
            elif output.dim() == 2:
                # (B, C) -> (B, 1, C)
                return output.unsqueeze(1)
    
    raise ValueError(f"Could not extract tokens from {backbone_name}, output shape: {output.shape}")

