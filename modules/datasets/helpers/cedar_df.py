import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from modules.datasets.helpers.constants import TRANSFORMS_PRE

"""
Download from https://www.cedar.buffalo.edu/NIJ/data/signatures.rar
Directory structure should look like this:
.
├── full_forg/
├── full_org/
└── Readme.txt
"""


DEFAULT_TEST_SIZE = 5 / 55  # 5 signers for testing, 55 - 5 = 50 for training
RANDOM_STATE = 339


def cedar_df(cedar_path, test_size=DEFAULT_TEST_SIZE):
    # Read from forgeries folder
    images_forged = glob.glob(os.path.join(cedar_path, "full_forg", "forgeries*.png"))
    images_forged = sorted(images_forged)

    # Read from originals folder
    images_original = glob.glob(os.path.join(cedar_path, "full_org", "original*.png"))
    images_original = sorted(images_original)

    forged_df = pd.DataFrame({"path": images_forged, "type": "forged"})
    forged_df["signer"] = forged_df["path"].str.extract(r"forgeries_(\d+)_").astype(int)

    original_df = pd.DataFrame({"path": images_original, "type": "original"})
    original_df["signer"] = (
        original_df["path"].str.extract(r"original_(\d+)_").astype(int)
    )

    # Combine into a single DataFrame
    cedar_df = pd.concat([original_df, forged_df], axis=0)

    # print(cedar_df.groupby(["signer", "type"]).count())

    # Perform a self join to get all combinations of samples for each signer
    cedar_df = pd.merge(cedar_df, cedar_df, on="signer", suffixes=("_first", "_second"))

    # Drop reflexive pairs
    cedar_df = cedar_df[cedar_df["path_first"] != cedar_df["path_second"]]

    # Drop forged-forged pairs (unused)
    cedar_df = cedar_df[
        ~((cedar_df["type_first"] == "forged") & (cedar_df["type_second"] == "forged"))
    ]

    # Temporarily a column of tuples to serve as a unique key for each pair
    cedar_df["pair"] = cedar_df[["path_first", "path_second"]].apply(
        lambda row: tuple(sorted(row)), axis=1
    )

    # Drop symmetric-duplicate pairs
    cedar_df = cedar_df.drop_duplicates(subset=["pair"], keep="first")

    # Discard temporary key (we don't need it anymore)
    cedar_df = cedar_df.drop("pair", axis=1)

    # Group-shuffle train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(cedar_df, groups=cedar_df["signer"]))

    cedar_df_train = cedar_df.iloc[train_idx]
    cedar_df_test = cedar_df.iloc[test_idx]

    # TODO ---------------------------------------------------------------------
    # Calculate stdev of all images
    PIL_images = [
        # "L" is luminance (grayscale) mode
        Image.open(path).convert("L")
        for path in cedar_df_train["path_first"].unique()
    ]
    transformed_images = [TRANSFORMS_PRE(image) for image in PIL_images]
    np_images = [np.array(image) for image in PIL_images]
    pixels = np.concatenate([image.flatten() for image in np_images])

    stdev = np.std(pixels)
    # --------------------------------------------------------------------------

    return cedar_df_train, cedar_df_test, stdev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
    args = parser.parse_args()

    cedar_df_train, cedar_df_test = cedar_df(args.cedar_path)

    print("Train dataset:")
    print(cedar_df_train.groupby(["signer", "type_first", "type_second"]).count())

    print("Test dataset:")
    print(cedar_df_test.groupby(["signer", "type_first", "type_second"]).count())
