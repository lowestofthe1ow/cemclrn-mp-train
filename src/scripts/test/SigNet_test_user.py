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
    default="checkpoints/finetuned/models/user1/model_2025-12-03 09:09:38.016142.pth",
    type=str,
    help="Path to trained model .ckpt file",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=15)
args = parser.parse_args()


TRAIN_STD = 0.2346486747264862

forged_path = "data/test_user_data/forged"
full_paths_forged = []
for entry in os.listdir(forged_path):
    full_path = os.path.join(forged_path, entry)
    full_paths_forged.append(full_path)

genuine_path = "data/test_user_data/genuine"
full_paths_genuine = []
i = 0
for entry in os.listdir(genuine_path):
    full_path = os.path.join(genuine_path, entry)
    full_paths_genuine.append(full_path)
    i = i + 1
    if i >= 5:
        break


df_pairs_genuine = pd.DataFrame(
    list(itertools.combinations(full_paths_genuine, 2)),
    columns=["orig", "not_orig"],
)
df_pairs_genuine["genuine"] = 1


df_pairs_forged = pd.DataFrame(
    list(itertools.product(full_paths_genuine, full_paths_forged)),
    columns=["orig", "not_orig"],
)
df_pairs_forged["genuine"] = 0

df_pairs_all = pd.concat([df_pairs_genuine, df_pairs_forged])

print()

print(df_pairs_all)

test_dataset = UserDataset(df_pairs_all, transform=TRANSFORMS_EVAL(TRAIN_STD))

test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)

model = SigNet()
state_dict = torch.load(args.ckpt_path)
model.load_state_dict(state_dict)


model.eval()  # TODO: Pretty sure Trainer.test() sets this already
trainer = pl.Trainer()
trainer.test(model, test_dataloader)
