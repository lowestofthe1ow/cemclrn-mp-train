import itertools
import mysql.connector as sql
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
from torchvision.transforms import InterpolationMode

from src.datasets.UserDataset import UserDataset
from src.engines.SigNet import SigNet
from src.utils.transforms.transforms import TRANSFORMS_TRAIN


USER_DATA_PATH = "data/user_data"


# Insert user data
def populate_signatures(db, cursor):
    base_dir = Path(USER_DATA_PATH).absolute()

    if base_dir.exists():
        print(f"User data directory accessed\n")

    for user_folder in sorted(base_dir.iterdir()):
        user_name = user_folder.name

        cursor.execute("INSERT IGNORE INTO users (name) VALUES (%s)", (user_name,))
        db.commit()

        cursor.execute("SELECT id FROM users WHERE name = %s", (user_name,))

        user_id = cursor.fetchone()[0]

        image_paths = sorted(user_folder.glob("*.png"))
        image_paths.extend(sorted(user_folder.glob("*.jpg")))
        image_paths.extend(sorted(user_folder.glob("*.jpeg")))

        i = 0
        for image_path in image_paths:
            image_path_str = os.path.join(user_folder, image_path)
            # Time the signature was added
            timestamp = os.path.getmtime(image_path_str)
            dt_timestamp = datetime.fromtimestamp(timestamp)
            sql_timestamp = dt_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # For checking image path and user
            # print(image_path_str + " from user " + str(user_id))
            cursor.execute(
                """
                        INSERT IGNORE INTO signatures (user_id, path, time_added)
                        VALUES (%s, %s, %s)
                        """,
                (user_id, image_path_str, sql_timestamp),
            )

            i += 1
            if i >= 15:
                break

    cursor.execute(
        """
                INSERT IGNORE INTO models (user_id, path, time_added)
                VALUES (%s, %s, %s)
                """,
        (
            user_id,
            "checkpoints/finetuned/models/user2/model_2025-12-03 18:07:54.710987.pth",
            sql_timestamp,
        ),
    )

    db.commit()
