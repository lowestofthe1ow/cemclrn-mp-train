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

from src.utils.transforms.transforms import TRANSFORMS_PRE

"""
Download from https://www.cedar.buffalo.edu/NIJ/data/signatures.rar
Directory structure should look like this:
.
├── full_forg/
├── full_org/
└── Readme.txt
"""

TOTAL_SIGNERS = 55
TEST_SIGNERS = 5
VAL_SIGNERS = 5
DEFAULT_TEST_SIZE = 5 / 55  # 5 signers for testing, 55 - 5 = 50 for training
RANDOM_STATE = 339


def cedar_df(cedar_path):
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
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIGNERS / TOTAL_SIGNERS,  # Default: 5 / 55
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(cedar_df, groups=cedar_df["signer"]))

    cedar_df_train_full = cedar_df.iloc[train_idx]
    cedar_df_test = cedar_df.iloc[test_idx]

    # Further group-shuffle split between train/validation
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SIGNERS / (TOTAL_SIGNERS - TEST_SIGNERS),  # Default: 5 / 50
        random_state=RANDOM_STATE,
    )
    train_idx, valid_idx = next(
        gss.split(cedar_df_train_full, groups=cedar_df_train_full["signer"])
    )

    cedar_df_valid = cedar_df_train_full.iloc[valid_idx]
    cedar_df_train = cedar_df_train_full.iloc[train_idx]

    # Calculate mean and stdev of all training set images
    PIL_images = [
        # "L" is luminance (grayscale) mode
        Image.open(path).convert("L")
        for path in cedar_df_train["path_first"].unique()
    ]
    # REQUIRED: Apply pre-transforms first
    # TODO: Further preprocessing
    transformed_images = [TRANSFORMS_PRE(image) for image in PIL_images]

    np_images = [np.array(image) for image in transformed_images]

    pixels = np.concatenate([image.flatten() for image in np_images])

    mean = np.mean(pixels)
    stdev = np.std(pixels)

    print("\nProcessed CEDAR dataset.\n")
    print(f"Training: {len(cedar_df_train)} images")
    print(f"Testing: {len(cedar_df_test)} images")
    print(f"Validation: {len(cedar_df_valid)} images\n")
    print(f"Mean: {mean}")
    print(f"Standard deviation: {stdev}")

    return cedar_df_train, cedar_df_test, cedar_df_valid, mean, stdev
