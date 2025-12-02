import numpy as np
import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F2
import torch.nn.functional as F


from src.utils.transforms.transforms import TRANSFORMS_EVAL
from src.datasets.process.cedar_df import cedar_df
from src.datasets.CEDARDataset import CEDARDataset
from src.engines.SigNet import SigNet

from PIL import Image

# Image standard dev. calculated during training for pixel values in [0, 1]
TRAIN_STD = 0.07225848734378815

# Threshold Euclidean distance to separate "genuine" vs. "forged" pairs
D_THRESHOLD = 0.6017965078353882

"""
TODO: This currently only checks a pair of images.
For a user, we want it to check the queried image x1 with all possible x2 in all
signatures for that user in the database.
"""

"""
def inference(model_path, x1_path, x2_path):

#    Args:
#        model_path: Path to model .pth file
#        x1:         Path to first image
#        x2:         Path to second image


    x1 = Image.open(x1_path).convert("L")
    x2 = Image.open(x2_path).convert("L")
    x1.show()

    transform = TRANSFORMS_EVAL(TRAIN_STD)

    x1 = transform(x1).unsqueeze(0)
    x2 = transform(x2).unsqueeze(0)

    state_dict = torch.load(model_path)

    model = SigNet()
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        y1, y2 = model(x1, x2)

        distance = F.pairwise_distance(y1, y2)
        prediction = distance <= D_THRESHOLD  # True if genuine pair

        return distance, prediction
"""

# Assuming x2_paths is a vector of paths
#  
def inference(model_path, x1_path, x2_paths):
    """
    Args:
        model_path: Path to model .pth file
        x1_path:         Path to first image
        x2_paths:         Path(s) to second image
    """

    x1 = Image.open(x1_path).convert("L")
    x1.show() # only for debugging

    transform = TRANSFORMS_EVAL(TRAIN_STD)

    x1 = transform(x1).unsqueeze(0)

    state_dict = torch.load(model_path)

    model = SigNet()
    model.load_state_dict(state_dict)
    model.eval()

    length = len(x2_paths)
    verdict = 0
    total_dist = 0

    for x in x2_paths:

        x2 = Image.open(x).convert("L")
        x2 = transform(x2).unsqueeze(0)

        with torch.no_grad():
            y1, y2 = model(x1, x2)

            distance = F.pairwise_distance(y1, y2).item()
            total_dist += distance
            prediction = distance <= D_THRESHOLD  # True if genuine pair

            if prediction:
                verdict += 1
            else:
                verdict -= 1

    total_dist /= length
    
    if verdict > 0:
        prediction = True
    else:
        prediction = False

    return total_dist, prediction

if __name__ == "__main__":
    distance, prediction = inference(
        "checkpoints/base_model2.pth",
        "data/user_data/user0/user0_3.jpg",
        "data/user_data/user0/user0_0.jpg",
    )

    print(f"Distance = {distance}, Prediction = {prediction}")

    """
    # For testing
    np.random.choice(my_list, size=3)

    files_ref = os.listdir("data/cedar/full_org")
    files_ref = random.ssampl
    files_query = os.listdir("data/user_data/user0")

    files_ref = [
        os.path.join("data/cedar/full_org", filename) for filename in files_ref
    ]
    files_query = [
        os.path.join("data/user_data/user0", filename) for filename in files_query
    ]

    print(files_ref)
    """
