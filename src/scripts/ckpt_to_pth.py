"""
Creates a smaller, more portable .pth file from a larger .ckpt file
"""

import argparse
import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode

from src.utils.transforms.transforms import TRANSFORMS_TRAIN, TRANSFORMS_EVAL
from src.datasets.process.cedar_df import cedar_df
from src.datasets.CEDARDataset import CEDARDataset
from src.engines.SigNet import SigNet

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str, help="Path to .ckpt input file")
parser.add_argument("--pth-path", type=str, help="Path to .pth output file")
args = parser.parse_args()

checkpoint = torch.load(args.ckpt_path, map_location="cpu")
state_dict = checkpoint["state_dict"]
torch.save(state_dict, args.pth_path)

print(f"Saved model weights from {args.ckpt_path} to {args.pth_path}")
