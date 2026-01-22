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
        name: Backbone identifier (mobilenet_v2, resnet18, resnet34, resnet50, clip_vit_b32)
        device: Device to load model on
        
    Returns:
        (model, feat_dim, preprocess_fn)
        - model: Frozen backbone in eval mode
        - feat_dim: Output feature dimension
        - preprocess_fn: Function that takes numpy RGB uint8 (H,W,3) -> torch tensor (1,3,H,W)
    """
    
    name = name.lower()
    
    # ImageNet normalization (used by torchvision models)
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
        
    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Supported: mobilenet_v2, resnet18, resnet34, resnet50, clip_vit_b32"
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
