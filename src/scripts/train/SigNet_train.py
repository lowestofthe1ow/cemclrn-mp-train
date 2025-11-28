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
parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=15)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()


# We don't use the mean, so we just skip it for now
train_df, _, valid_df, _, stdev = cedar_df(args.cedar_path)

train_dataset = CEDARDataset(train_df, TRANSFORMS_TRAIN(stdev=stdev))
val_dataset = CEDARDataset(valid_df, TRANSFORMS_EVAL(stdev=stdev))

# Set up data loaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
)

model = SigNet()


os.makedirs("checkpoints", exist_ok=True)
logger = TensorBoardLogger("tb_logs", name="cedar")

# Use when using early stopping
# early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",  # Directory to save checkpoints
    filename="fixed_thresh_divide_{epoch:02d}-{val_loss:.2f}",  # Custom filename pattern
    monitor="val_loss",  # Metric to monitor for saving the best model
    mode="min",  # Save when the monitored metric is minimized
    save_top_k=3,  # Keep only the best 'k' checkpoints
)

trainer = pl.Trainer(
    default_root_dir="checkpoints",
    logger=logger,
    min_epochs=args.epochs,
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloader, val_dataloader)
