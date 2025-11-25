# https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

import argparse
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from modules.datasets.helpers.cedar_df import cedar_df


class CEDARDataset(Dataset):
    """CEDAR signatures dataset"""

    def __init__(self, cedar_df, transform):
        self.transform = transform
        self.cedar_df = cedar_df

    def __len__(self):
        return len(self.cedar_df)

    def __getitem__(self, idx):
        data = self.cedar_df.iloc[idx]
        x1 = self.transform(Image.open(data.path_first).convert("L"))
        x2 = self.transform(Image.open(data.path_second).convert("L"))

        if data.type_first == "original" and data.type_second == "original":
            return x1, x2, 1
        elif data.type_first == "forged" and data.type_second == "original":
            return x1, x2, 0
        elif data.type_first == "original" and data.type_second == "forged":
            return x1, x2, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
    args = parser.parse_args()

    train_df, test_df, stdev = cedar_df(args.cedar_path)

    print(f"Loaded CEDAR dataset and calculated stdev to be {stdev}")

    transform = transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.PILToTensor(),  # Preserves [0, 255] scale
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )

    dataset = CEDARDataset(train_df, transform)
    print(dataset.__getitem__(0))
