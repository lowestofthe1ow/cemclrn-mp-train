import itertools
import mysql.connector as sql
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split
from torchvision.transforms import InterpolationMode

from src.datasets.process.cedar_df import cedar_df
from src.datasets.UserDataset import UserDataset
from src.engines.SigNet import SigNet
from src.utils.transforms.transforms import TRANSFORMS_TRAIN, TRANSFORMS_EVAL


TRAIN_STD = 0.2346486747264862

# SQL for user signatures
db = sql.connect(
    host="localhost",  # change if needed
    user="root",  # change if needed
    password="fujita_kotone",  # change if needed
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


def makeDf(repeat, user_id):
    query = """
    SELECT path
    FROM signatures 
    WHERE user_id = %s
    ORDER BY time_added DESC 
    LIMIT 15
    """

    train_df, _, valid_df, _, stdev = cedar_df("data/cedar")

    user_sign_df = pd.read_sql(query, db, params=(user_id,))

    # Genuine-fake pairs
    fake_sign_df = user_sign_df.loc[user_sign_df.index.repeat(repeat)].reset_index(
        drop=True
    )

    print(train_df["path_first"].unique())

    random_orgs = np.random.choice(
        train_df["path_first"].unique(), size=15 * repeat, replace=False
    )

    fake_sign_df = fake_sign_df.rename(columns={"path": "orig"})
    fake_sign_df["not_orig"] = random_orgs
    fake_sign_df["genuine"] = 0

    fake_sign_df = fake_sign_df.rename(columns={"path": "orig"})

    # Genuine-genuine pairs
    pairs = list(itertools.combinations(user_sign_df["path"], 2))

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


def finetune(user_id, batch_size=32, num_workers=4):
    # Check if correct
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    print(f"Total users in database: {total_users}")
    cursor.execute("SELECT COUNT(*) FROM signatures")
    total_signatures = cursor.fetchone()[0]
    print(f"Total signatures in database: {total_signatures}")

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", None)

    user_df = makeDf(2, user_id)

    train_df, val_df = train_test_split(
        user_df,
        train_size=int(len(user_df) * 0.9),
        random_state=339,
        shuffle=True,
    )

    train_dataset = UserDataset(train_df, TRANSFORMS_TRAIN(stdev=TRAIN_STD))
    val_dataset = UserDataset(val_df, TRANSFORMS_EVAL(stdev=TRAIN_STD))

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

    # state_dict = torch.load("checkpoints/base_model.pth")

    # model = SigNet()
    # model.load_state_dict(state_dict)

    model = SigNet.load_from_checkpoint(
        "checkpoints/FINAL_epoch=10-val_loss=0.05998.ckpt",
        learning_rate=1e-6,
    )
    # model.eval()

    for param in model.cnn.features.parameters():
        param.requires_grad = False

    for param in model.cnn.features[14].parameters():
        param.requires_grad = True

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

    full_path = os.path.join(model_path, model_filename)

    torch.save(state_dict, full_path)

    return full_path


if __name__ == "__main__":
    finetune(1)
