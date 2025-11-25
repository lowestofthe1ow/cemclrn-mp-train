import argparse
import os
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from modules.datasets.helpers.cedar_df import cedar_df
from modules.datasets.torch.CEDARDataset import CEDARDataset
from modules.models.SigNetSiamese import SigNetSiamese

parser = argparse.ArgumentParser()
parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
parser.add_argument("--ckpt-path", type=str, help="Path to trained model .ckpt file")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=15)
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

BATCH_SIZE = 16

train_df, test_df, stdev = cedar_df(args.cedar_path)

print(f"Loaded CEDAR dataset and calculated stdev to be {stdev}")
print(f"Test dataset size: {len(test_df)}")

transform = transforms.Compose(
    [
        transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
        transforms.RandomInvert(p=1.0),
        transforms.PILToTensor(),
        tensor.to(torch.float)
        # Divide by stdev but don't subtract by a mean value
        transforms.Normalize(mean=0, std=stdev),
    ]
)

test_dataset = CEDARDataset(test_df, transform)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, num_workers=args.num_workers
)

model = SigNetSiamese.load_from_checkpoint(args.ckpt_path)
trainer = pl.Trainer()
trainer.test(model, test_dataloader)
