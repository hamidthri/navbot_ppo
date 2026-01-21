#!/usr/bin/env python3
"""
Vision feature encoder using pretrained MobileNetV2.
Outputs fixed-size feature vector for concatenation with other observations.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class VisionEncoder(nn.Module):
    """
    Pretrained vision encoder that extracts features from RGB images.
    Uses frozen MobileNetV2 backbone + small projection to output_dim.
    """
    def __init__(self, output_dim=64, use_pretrained=True):
        super(VisionEncoder, self).__init__()
        self.output_dim = output_dim
        
        # Load pretrained MobileNetV2
        print(f"[VisionEncoder] Loading MobileNetV2 (pretrained={use_pretrained})...", flush=True)
        try:
            mobilenet = models.mobilenet_v2(pretrained=use_pretrained)
            print(f"[VisionEncoder] MobileNetV2 loaded successfully", flush=True)
        except Exception as e:
            print(f"[VisionEncoder] ERROR loading pretrained weights: {e}", flush=True)
            print(f"[VisionEncoder] Falling back to random initialization", flush=True)
            mobilenet = models.mobilenet_v2(pretrained=False)
        
        # Extract feature extractor (remove classifier)
        self.features = mobilenet.features
        
        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False
        
        # MobileNetV2 outputs 1280-dim features after adaptive pooling
        # Add projection head to reduce to output_dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, output_dim),
            nn.LayerNorm(output_dim)  # Normalize for stable scale
        )
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.eval()  # Set to eval mode (frozen)
        print(f"[VisionEncoder] Initialized with output_dim={output_dim}", flush=True)
    
    def preprocess_image(self, image_msg_or_array):
        """
        Convert ROS Image message or numpy array to preprocessed tensor.
        Args:
            image_msg_or_array: sensor_msgs/Image or numpy array (H, W, 3) in BGR or RGB
        Returns:
            torch.Tensor of shape (1, 3, 224, 224)
        """
        # Handle ROS Image message
        if hasattr(image_msg_or_array, 'encoding'):
            # Convert ROS message to numpy manually to avoid cv_bridge
            import numpy as np
            height = image_msg_or_array.height
            width = image_msg_or_array.width
            encoding = image_msg_or_array.encoding
            
            # Convert bytes to numpy array
            img_data = np.frombuffer(image_msg_or_array.data, dtype=np.uint8)
            
            if encoding == 'rgb8':
                cv_image = img_data.reshape((height, width, 3))
            elif encoding == 'bgr8':
                cv_image = img_data.reshape((height, width, 3))
                # Swap BGR to RGB
                cv_image = cv_image[:, :, [2, 1, 0]]
            else:
                # Fallback: assume RGB
                cv_image = img_data.reshape((height, width, 3))
        else:
            # Assume numpy array in BGR, convert to RGB
            cv_image = image_msg_or_array
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                # Assume BGR, swap to RGB
                cv_image = cv_image[:, :, [2, 1, 0]]
        
        # Apply transforms
        tensor = self.transform(cv_image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def forward(self, image_tensor):
        """
        Extract features from preprocessed image tensor.
        Args:
            image_tensor: torch.Tensor of shape (B, 3, 224, 224)
        Returns:
            torch.Tensor of shape (B, output_dim)
        """
        with torch.no_grad():
            x = self.features(image_tensor)
            x = self.projection(x)
        return x
    
    def encode_image(self, image_msg_or_array, device='cpu'):
        """
        End-to-end encoding: preprocess + extract features.
        Returns numpy array of shape (output_dim,)
        """
        tensor = self.preprocess_image(image_msg_or_array).to(device)
        features = self.forward(tensor)
        features_np = features.cpu().numpy()[0]
        
        # Log stats once
        if not hasattr(self, '_logged_stats'):
            self._logged_stats = True
            print(f"[VisionEncoder] Feature stats: shape={features_np.shape}, "
                  f"min={features_np.min():.4f}, mean={features_np.mean():.4f}, "
                  f"max={features_np.max():.4f}, norm={np.linalg.norm(features_np):.4f}, "
                  f"has_nan={np.isnan(features_np).any()}", flush=True)
        
        return features_np


def test_vision_encoder():
    """Test the vision encoder with a dummy image"""
    print("Testing VisionEncoder...")
    encoder = VisionEncoder(output_dim=64, use_pretrained=True)
    
    # Create dummy RGB image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Encode
    features = encoder.encode_image(dummy_image)
    print(f"Output features shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Feature mean: {features.mean():.4f}, std: {features.std():.4f}")
    print("VisionEncoder test passed!")


if __name__ == '__main__':
    test_vision_encoder()
