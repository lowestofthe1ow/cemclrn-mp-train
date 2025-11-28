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

        # We define the CNN directly based on Table 1 of the SigNet paper:
        # https://arxiv.org/pdf/1707.02131

        self.features = nn.Sequential(
            # ------------------------------------------------------------------
            # (1) Convolution 96 × 11 × 11 (stride = 1) + ReLU
            nn.Conv2d(
                in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.ReLU(inplace=True),
            # ------------------------------------------------------------------
            # (2) Batch Norm. (ϵ = 10^-6, momentum = 0.1 from Keras 0.9)
            # nn.LazyBatchNorm2d(eps=1e-05, momentum=0.1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # ------------------------------------------------------------------
            # (3) Pooling 96 × 3 × 3 (stride = 2)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # ------------------------------------------------------------------
            # (4) Convolution 256 × 5 × 5 (stride = 1, pad = 2) + ReLU
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            # ------------------------------------------------------------------
            # (5) Batch Norm. (ϵ = 10^-6, momentum = 0.1 from Keras 0.9)
            # nn.LazyBatchNorm2d(eps=1e-05, momentum=0.1),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # ------------------------------------------------------------------
            # (6) Pooling 256 × 3 × 3 (stride = 2) + Dropout (p = 0.3)
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.3),
            # ------------------------------------------------------------------
            # (7) Convolution 384 × 3 × 3 (stride = 1, pad = 1) + ReLU
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # ------------------------------------------------------------------
            # (8) Convolution 256 × 3 × 3 (stride = 1, pad = 1) + ReLU
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            # ------------------------------------------------------------------
            # (9) Pooling 256 × 3 × 3 (stride = 2) + Dropout (p = 0.3) + Flatten
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.3),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            # ------------------------------------------------------------------
            # (10) FC (1024) + ReLU + Dropout (p=0.5)
            # 108800 = 17 * 25 * 256
            nn.Linear(3840, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # ------------------------------------------------------------------
            # (11) FC (OUTPUT_CLASSES) + ReLU
            nn.Linear(1024, OUTPUT_CLASSES),
            nn.ReLU(inplace=True),
        )

        self.features.apply(initialize_weights)
        self.classifier.apply(initialize_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
