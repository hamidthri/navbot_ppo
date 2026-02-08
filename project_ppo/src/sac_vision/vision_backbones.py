#!/usr/bin/env python3
"""
Vision Backbones for SAC with ImageNet Normalization
Supports ResNet-18, ResNet-50, and other architectures
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for visual feature extraction with ImageNet normalization.
    Supports ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
    """
    def __init__(self, architecture='resnet18', pretrained=True, output_dim=512):
        super(ResNetBackbone, self).__init__()
        self.architecture = architecture
        self.output_dim = output_dim
        
        # ImageNet normalization (CRITICAL for pretrained ResNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Load pretrained ResNet
        if architecture == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            backbone_output_dim = 512
        elif architecture == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            backbone_output_dim = 512
        elif architecture == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            backbone_output_dim = 2048
        elif architecture == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            backbone_output_dim = 2048
        elif architecture == 'resnet152':
            resnet = models.resnet152(pretrained=pretrained)
            backbone_output_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Remove the final FC layer
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # Add projection layer
        self.projection = nn.Linear(backbone_output_dim, output_dim)
        
        print(f"[ResNetBackbone] {architecture}, output_dim={output_dim}, pretrained={pretrained}")
        
    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, C, H, W) - expects values in [0, 1]
            
        Returns:
            features: Visual features (B, output_dim)
        """
        # Apply ImageNet normalization
        x = self.normalize(x)
        
        # Extract features
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


class SimpleCNN(nn.Module):
    """
    Simple CNN backbone for baseline comparison
    """
    def __init__(self, output_dim=512):
        super(SimpleCNN, self).__init__()
        self.output_dim = output_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
        
        print(f"[SimpleCNN] output_dim={output_dim}")
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_vision_backbone(architecture='resnet18', pretrained=True, output_dim=512):
    """
    Factory function to get vision backbone
    """
    if 'resnet' in architecture.lower():
        return ResNetBackbone(architecture=architecture, pretrained=pretrained, output_dim=output_dim)
    elif architecture.lower() == 'simple_cnn':
        return SimpleCNN(output_dim=output_dim)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")