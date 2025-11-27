import argparse
import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from modules.datasets.helpers.constants import TRANSFORMS_TRAIN
from modules.datasets.helpers.cedar_df import cedar_df
from modules.datasets.torch.CEDARDataset import CEDARDataset
from modules.models.SigNetSiamese import SigNetSiamese


parser = argparse.ArgumentParser()
parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=15)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

train_df, test_df, stdev = cedar_df(args.cedar_path)

print(f"Loaded CEDAR dataset and calculated stdev to be {stdev}")

train_dataset = CEDARDataset(train_df, TRANSFORMS_TRAIN(stdev))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)

model = SigNetSiamese()

os.makedirs("checkpoints", exist_ok=True)

logger = TensorBoardLogger("tb_logs", name="cedar")
trainer = pl.Trainer(
    default_root_dir="checkpoints",
    logger=logger,
    min_epochs=args.epochs,
    max_epochs=args.epochs,
)

trainer.fit(model, train_dataloader)
