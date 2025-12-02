# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

import argparse
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.utils.transforms.transforms import TRANSFORMS_TRAIN
from src.datasets.process.cedar_df import cedar_df


class UserDataset(Dataset):
    """CEDAR signatures dataset"""

    def __init__(self, user_df, transform):
        self.transform = transform
        self.user_df = user_df  # SHOULD BE TEST SET

    def __len__(self):
        return len(self.user_df)

    def __getitem__(self, idx):
        data = self.user_df.iloc[idx]
        x1 = self.transform(Image.open(data.path_first).convert("L"))
        x2 = self.transform(Image.open(data.path_second).convert("L"))

        """
        Uncomment to show image output
        torch.set_printoptions(profile="full")
        print(x1)
        F.to_pil_image(x1 / torch.max(x1)).show()
        quit()
        """

        return x1, x2, 0
