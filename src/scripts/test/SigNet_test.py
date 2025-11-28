import argparse
import os
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from src.utils.transforms.transforms import TRANSFORMS_EVAL
from src.datasets.process.cedar_df import cedar_df
from src.datasets.CEDARDataset import CEDARDataset
from src.engines.SigNet import SigNet

parser = argparse.ArgumentParser()
parser.add_argument("--cedar-path", type=str, help="Path to CEDAR dataset folder")
parser.add_argument("--ckpt-path", type=str, help="Path to trained model .ckpt file")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=15)
args = parser.parse_args()


train_df, test_df, valid_df, mean, stdev = cedar_df(args.cedar_path)

# print(f"Test dataset size: {len(test_df)}")

test_dataset = CEDARDataset(test_df, TRANSFORMS_EVAL(stdev=stdev))
test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
)

model = SigNet.load_from_checkpoint(args.ckpt_path)


model.eval()  # TODO: Pretty sure Trainer.test() sets this already
trainer = pl.Trainer()
trainer.test(model, test_dataloader)
