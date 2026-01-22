"""
Frozen MobileNetV2 backbone for vision feature extraction.
Used during rollout to compute 1280-d features ONCE per step.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenMobileNetBackbone(nn.Module):
    """Frozen MobileNetV2 backbone for extracting 1280-d features"""
    
    def __init__(self, use_pretrained=True):
        super(FrozenMobileNetBackbone, self).__init__()
        
        # Load MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=use_pretrained)
        
        # Extract feature extractor (everything before classifier)
        self.features = mobilenet.features
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.eval()  # Always in eval mode
        
        # Verify frozen
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[VisionBackbone] MobileNetV2 backbone: {total:,} params, {trainable:,} trainable (FROZEN)", flush=True)
        assert trainable == 0, "Backbone must be frozen!"
    
    def preprocess_image(self, img_array):
        """
        Preprocess numpy uint8 HxWx3 RGB image to tensor.
        Returns: (1, 3, 224, 224) tensor
        """
        if img_array is None:
            # Return zeros if no image
            return torch.zeros(1, 3, 224, 224)
        
        # Convert to tensor and normalize
        img_tensor = self.transform(img_array).unsqueeze(0)  # (1, 3, 224, 224)
        return img_tensor
    
    @torch.no_grad()
    def forward(self, x):
        """
        Extract 1280-d features from image tensor.
        Input: (B, 3, 224, 224)
        Output: (B, 1280)
        """
        x = self.features(x)  # (B, 1280, 7, 7)
        x = self.avgpool(x)   # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1280)
        return x
    
    @torch.no_grad()
    def encode_image(self, img_array):
        """
        End-to-end: numpy image -> 1280-d feature vector.
        Returns: numpy array (1280,)
        """
        img_tensor = self.preprocess_image(img_array).to(device)
        features = self.forward(img_tensor)
        return features.cpu().numpy()[0]
