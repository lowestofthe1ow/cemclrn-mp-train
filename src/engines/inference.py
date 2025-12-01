import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from src.utils.transforms.transforms import TRANSFORMS_EVAL
from src.datasets.process.cedar_df import cedar_df
from src.datasets.CEDARDataset import CEDARDataset
from src.engines.SigNet import SigNet

from PIL import Image

# Image standard dev. calculated during training for pixel values in [0, 1]
TRAIN_STD = 0.07225848734378815

# Threshold Euclidean distance to separate "genuine" vs. "forged" pairs
D_THRESHOLD = 0.20001120865345

"""
TODO: This currently only checks a pair of images.
For a user, we want it to check the queried image x1 with all possible x2 in all
signatures for that user in the database.
"""


def inference(model_path, x1_path, x2_path):
    """
    Args:
        model_path: Path to model .pth file
        x1:         Path to first image
        x2:         Path to second image
    """

    x1 = Image.open(x1_path).convert("L")
    x2 = Image.open(x2_path).convert("L")

    transform = TRANSFORMS_EVAL(TRAIN_STD)

    x1 = transform(x1).unsqueeze(0)
    x2 = transform(x2).unsqueeze(0)

    state_dict = torch.load(model_path)

    model = SigNet()
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        y1, y2 = model(x1, x2)

        distance = F.pairwise_distance(y1, y2)
        prediction = distance <= D_THRESHOLD  # True if genuine pair


if __name__ == "__main__":
    # For testing
    inference(
        "checkpoints/model_base.pth",
        "data/cedar/full_org/original_1_1.png",
        "data/cedar/full_org/original_1_2.png",
    )
