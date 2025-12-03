import argparse
import os
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
import itertools
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from src.utils.transforms.transforms import TRANSFORMS_EVAL
from src.datasets.process.cedar_df import cedar_df
from src.datasets.UserDataset import UserDataset
from src.datasets.CEDARDataset import CEDARDataset
from src.engines.SigNet import SigNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cedar-path", type=str, default="data/cedar", help="Path to CEDAR dataset folder"
)
parser.add_argument(
    "--ckpt-path",
    default="checkpoints/finetuned/models/user1/model_2025-12-03 11:31:50.913674.pth",
    type=str,
    help="Path to trained model .ckpt file",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=15)
args = parser.parse_args()


TRAIN_STD = 0.2346486747264862

EXPECTED = 0

QUERY_PATH_REAL = "data/test_user_data/genuine/user1_16.jpg"

full_paths_genuine = []
genuine_path = "data/user_data/user1"
i = 0
for entry in os.listdir(genuine_path):
    full_path = os.path.join(genuine_path, entry)
    full_paths_genuine.append(full_path)
    i += 1
    if i >= 15:
        break

testing_df1 = pd.DataFrame(
    {"not_orig": QUERY_PATH_REAL, "orig": full_paths_genuine, "genuine": 1}
)


QUERY_PATH_FORGED = "data/test_user_data/forged/20251202_114735.jpg"
i = 0
full_paths_forged = []
forged_path = "data/user_data/user1"
for entry in os.listdir(forged_path):
    full_path = os.path.join(forged_path, entry)
    full_paths_forged.append(full_path)
    i += 1
    if i >= 15:
        break

testing_df2 = pd.DataFrame(
    {"not_orig": QUERY_PATH_FORGED, "orig": full_paths_genuine, "genuine": 0}
)

testing_df = pd.concat([testing_df1, testing_df2])

test_dataset = UserDataset(testing_df, transform=TRANSFORMS_EVAL(TRAIN_STD))

test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)

model = SigNet()
state_dict = torch.load(args.ckpt_path)
model.load_state_dict(state_dict)


model.eval()  # TODO: Pretty sure Trainer.test() sets this already
trainer = pl.Trainer()
trainer.test(model, test_dataloader)
