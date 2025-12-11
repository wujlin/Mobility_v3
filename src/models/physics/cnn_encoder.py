import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Encodes local Navigation Patch (3, K, K) into a vector.
    Channels: Direction-Y, Direction-X, Speed.
    """
    def __init__(self, in_channels=3, output_dim=32, patch_size=32):
        super().__init__()
        
        # Simple 3-layer CNN
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 8x8
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 4x4
            
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        return self.net(x)
