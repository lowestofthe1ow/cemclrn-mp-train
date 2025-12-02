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

from torch.utils.mobile_optimizer import optimize_for_mobile

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str, help="Path to .ckpt input file")
parser.add_argument("--pth-path", type=str, help="Path to .pth output file")
args = parser.parse_args()

checkpoint = torch.load(args.ckpt_path, map_location="cpu")
state_dict = checkpoint["state_dict"]
torch.save(state_dict, args.pth_path)

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):  # <--- ADJUST THIS PREFIX based on your SigNet
        new_state_dict[k[6:]] = v  # Remove the 'model.' prefix
    elif k.startswith("cnn."):  # <--- Check for other common prefixes
        new_state_dict[k[4:]] = v  # Remove the 'net.' prefix
    else:
        new_state_dict[k] = v

model = SigNet().cnn
model.load_state_dict(new_state_dict)
model.eval()

scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("finetuned_model2.ptl")

print(f"Saved model weights from {args.ckpt_path} to {args.pth_path}")
