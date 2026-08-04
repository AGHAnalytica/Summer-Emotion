import torch
from torch import nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes), )

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2),)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x




if __name__ == "__main__":
    model = EmotionCNN()

    images = torch.randn(4, 1, 256, 256)
    predictions = model(images)

    print("Input:", images.shape)
    print("Output:", predictions.shape)



"""
[batch, 1, 256, 256]
↓
[batch, 32, 128, 128]
↓
[batch, 64, 64, 64]
↓
[batch, 128, 32, 32]
↓
[batch, 256, 16, 16]
↓ AdaptiveAvgPool
[batch, 256, 1, 1]
↓
[batch, 7]
"""