import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import numpy as np

import matplotlib.pyplot as plt  #
import seaborn as sns

from src.models.SigNetCNN import SigNetCNN

MARGIN = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4  # 0.0005
MOMENTUM = 0.9
FUZZY = 1e-8
GAMMA = 0.1


# See https://github.com/VinhLoiIT/signet-pytorch/blob/master/model.py
# Based on Eq. 1 of SigNet paper (https://arxiv.org/pdf/1707.02131)
# Values for α and β are not mentioned, so we use 1
def contrastive_loss(output1, output2, y):
    euclidean_distance = F.pairwise_distance(output1, output2)
    contrastive_loss = y * euclidean_distance**2 + (1 - y) * (
        torch.max(torch.zeros_like(euclidean_distance), MARGIN - euclidean_distance)
        ** 2
    )
    contrastive_loss = torch.mean(contrastive_loss, dtype=torch.float)

    return contrastive_loss, euclidean_distance


class SigNet(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # "Branch" CNN of the Siamese network
        self.cnn = SigNetCNN()

        self.test_distances = []
        self.test_y = []

    def forward(self, x1, x2):
        y1 = self.cnn(x1)
        y2 = self.cnn(x2)
        return y1, y2

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self(x1, x2)
        loss, _ = contrastive_loss(output1, output2, y)

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self(x1, x2)
        loss, _ = contrastive_loss(output1, output2, y)

        # Log validation loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, y = batch
        output1, output2 = self(x1, x2)

        loss, distance = contrastive_loss(output1, output2, y)

        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.test_distances.append(distance.cpu().numpy())
        self.test_y.append(y.cpu().numpy())

    def on_test_epoch_end(self):
        np_distances = np.concatenate(self.test_distances)
        distances = torch.from_numpy(np_distances)
        y = torch.from_numpy(np.concatenate(self.test_y))

        zeros = np.zeros_like(np_distances)
        plt.figure(figsize=(12, 4))
        sns.scatterplot(
            x=np_distances, y=zeros, hue=y, palette="viridis", s=50, legend=False
        )
        plt.title("Euclidean distance distribution for CEDAR test set", fontsize=14)
        plt.xlabel("Euclidean distance", fontsize=12)
        plt.ylim(-0.1, 0.1)
        plt.show()

        min_threshold_d = min(distances)
        max_threshold_d = max(distances)
        max_acc = 0
        same_id = y == 1

        best_threshold_d = min_threshold_d

        # Search for an optimal threshold d that separates predicted genuine pairs from predicted forged pairs
        for threshold_d in torch.arange(min_threshold_d, max_threshold_d + 0.1, 0.1):
            true_positive = (distances <= threshold_d) & (same_id)
            true_positive_rate = true_positive.sum().float() / same_id.sum().float()
            true_negative = (distances > threshold_d) & (~same_id)
            true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

            acc = 0.5 * (true_negative_rate + true_positive_rate)

            if acc > max_acc:
                max_acc = acc
                best_threshold_d = threshold_d

        self.log("max_accuracy", max_acc, prog_bar=True)
        self.log("optimal_threshold", best_threshold_d)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.cnn.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
            eps=FUZZY,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)  # Divide by 10

        """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[14, 18],
            gamma=0.1,  # Divide by 10
        )
        """

        return optimizer
