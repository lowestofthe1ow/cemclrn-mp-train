import os
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import mysql.connector as sql
from pathlib import Path
from torch.utils.data import random_split
import torch
import itertools

import pytorch_lightning as pl
import torchvision.transforms as transforms

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode

from src.engines.SigNet import SigNet
from src.datasets.UserDataset import UserDataset
from src.utils.transforms.transforms import TRANSFORMS_TRAIN
from torch.utils.data import DataLoader
from datetime import datetime

TRAIN_STD = 0.07225848734378815

# SQL for user signatures
db = sql.connect(
    host="localhost",  # change if needed
    user="root",  # change if needed
    password="ManCC75?$@",  # change if needed
    database="signatures",  # change if needed
)
cursor = db.cursor()
user_data_path = "data/user_data"
cedar_org_path = "data/cedar/full_org"


# Create tables
def create_tables(cursor):
    cursor.execute("DELETE FROM signatures")
    cursor.execute("DELETE FROM users")

    # Reset auto-increment counters
    cursor.execute("ALTER TABLE users AUTO_INCREMENT = 1")
    cursor.execute("ALTER TABLE signatures AUTO_INCREMENT = 1")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS signatures (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            path VARCHAR(255) NOT NULL,
            time_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
            REFERENCES users(id)
        )
        """
    )
    db.commit()


# Insert user data
def insert_user_data(cursor):
    create_tables(cursor)
    base_dir = Path(user_data_path).absolute()
    if base_dir.exists():
        print(f"User data directory accessed\n")
    else:
        print("Cry")

    for user_folder in sorted(base_dir.iterdir()):
        user_name = user_folder.name

        cursor.execute("INSERT IGNORE INTO users (name) VALUES (%s)", (user_name,))
        db.commit()

        cursor.execute("SELECT id FROM users WHERE name = %s", (user_name,))

        user_id = cursor.fetchone()[0]

        image_paths = sorted(user_folder.glob("*.png"))
        image_paths.extend(sorted(user_folder.glob("*.jpg")))
        image_paths.extend(sorted(user_folder.glob("*.jpeg")))

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
        db.commit()


def makeDf(repeat, user_id):
    query = """
    SELECT path
    FROM signatures 
    WHERE user_id = %s
    ORDER BY time_added DESC 
    LIMIT 15
    """

    user_sign_df = pd.read_sql(query, db, params=(user_id,))

    # Genuine-fake pairs
    fake_sign_df = user_sign_df.loc[user_sign_df.index.repeat(repeat)].reset_index(
        drop=True
    )

    all_org = [
        os.path.join(cedar_org_path, f)
        for f in os.listdir(cedar_org_path)
        if os.path.isfile(os.path.join(cedar_org_path, f))
    ]
    random_orgs = np.random.choice(all_org, size=15 * repeat, replace=False)

    fake_sign_df = fake_sign_df.rename(columns={"path": "orig"})
    fake_sign_df["not_orig"] = random_orgs
    fake_sign_df["genuine"] = 0

    fake_sign_df = fake_sign_df.rename(columns={"path": "orig"})

    # Genuine-genuine pairs
    pairs = list(itertools.permutations(user_sign_df["path"], 2))

    gen_sign_df = pd.DataFrame(pairs, columns=["path", "not_orig"])
    gen_sign_df = gen_sign_df.rename(columns={"path": "orig"})
    gen_sign_df["genuine"] = 1

    combined_df = pd.concat([fake_sign_df, gen_sign_df]).reset_index(drop=True)

    """ Uncomment to test
    print("Genuine-fake pairs ======================================================")
    print(f"Number of unique elements in orig: {fake_sign_df['orig'].nunique()}")
    print(f"Number of unique elements in not_orig: {fake_sign_df['not_orig'].nunique()}")
    print(fake_sign_df)
    print("Genuine-ganuine pairs ======================================================")
    print(f"Number of unique elements in orig: {gen_sign_df['orig'].nunique()}")
    print(f"Number of unique elements in not_orig: {gen_sign_df['not_orig'].nunique()}")
    print(gen_sign_df)
    print("\nCombined pairs ======================================================")
    print(combined_df)
    """

    return combined_df


def finetune(user_id, batch_size=32, num_workers=15):
    insert_user_data(cursor)

    # Check if correct
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    print(f"Total users in database: {total_users}")
    cursor.execute("SELECT COUNT(*) FROM signatures")
    total_signatures = cursor.fetchone()[0]
    print(f"Total signatures in database: {total_signatures}")

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    user_df = makeDf(14, user_id)

    full_dataset = UserDataset(user_df, TRANSFORMS_TRAIN(stdev=TRAIN_STD))

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [
            int(0.9 * len(full_dataset)),
            len(full_dataset) - int(0.9 * len(full_dataset)),
        ],
        generator=torch.Generator().manual_seed(339),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    state_dict = torch.load("checkpoints/base_model2.pth")

    model = SigNet()
    model.load_state_dict(state_dict)

    for param in model.cnn.features.parameters():
        param.requires_grad = False

    logger = TensorBoardLogger("tb_logs", name="cedar")

    # Use when using early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/finetuned/checkpoints/",  # Directory to save checkpoints
        filename=f"user{user_id}_{{epoch:02d}}_{{val_loss:.2f}}",  # Custom filename pattern
        monitor="val_loss",  # Metric to monitor for saving the best model
        mode="min",  # Save when the monitored metric is minimized
        save_top_k=1,  # Keep only the best checkpoint
    )

    trainer = pl.Trainer(
        default_root_dir="checkpoints",
        logger=logger,
        min_epochs=0,
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    model_path = f"checkpoints/finetuned/models/user{user_id}/"

    # TODO: Add to an SQL table
    model_filename = f"model_{datetime.now()}.pth"

    os.makedirs(model_path, exist_ok=True)

    torch.save(state_dict, os.path.join(model_path, model_filename))


if __name__ == "__main__":
    finetune(1)
