import torch
import torch.nn as nn

OUTPUT_CLASSES = 128


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # Paper seems to have used Xavier uniform initialization
        nn.init.xavier_uniform_(m.weight)
        # Biases are initialized to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SigNetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # We define the backbone CNN directly based on Table II of this paper:
        # https://ieeexplore.ieee.org/document/11078027
        # This is a more modern approach vs. the original SigNet

        self.features = nn.Sequential(
            # ------------------------------------------------------------------
            # (1) Convolution + Leaky ReLU + Pooling
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ------------------------------------------------------------------
            # (2) Convolution + Batch norm. + Leaky ReLU + Pooling
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ------------------------------------------------------------------
            # (3) Convolution + Batch norm. + Leaky ReLU + Pooling
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # ------------------------------------------------------------------
            # (4) Convolution + Leaky ReLU + Dropout
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.3),  # Rate: 0.2
            # ------------------------------------------------------------------
            # (5) Convolution + Leaky ReLU + Pooling + Dropout
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(p=0.3),  # Rate: 0.2
            # ------------------------------------------------------------------
            # (6) Global avg. pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            # ------------------------------------------------------------------
            # (7) Dense
            nn.Dropout(p=0.6),  # Current best has this at 0.5
            nn.Linear(in_features=512, out_features=512),
            # ------------------------------------------------------------------
            # (8) Dense
            nn.Dropout(p=0.6),  # Current best has this at 0.5
            nn.Linear(in_features=512, out_features=128),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
