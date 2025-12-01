import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2

from PIL import Image
from torchvision.transforms import InterpolationMode

from src.utils.transforms.OtsuRemoveNoise import OtsuRemoveNoise

TRANSFORMS_PRE = transforms.Compose(
    [
        transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
        transforms.RandomInvert(p=1.0),
        transforms.ToTensor(),
        # OtsuBinarize(), TODO: Binarization causes rapid overfitting
    ]
)

"""
TODO: Add a random affine transform to train for better generalization? Idk
transforms.RandomAffine(
    degrees=10,
    scale=(0.9, 1.1),
    interpolation=InterpolationMode.BILINEAR,
    fill=0,
),
"""


def TRANSFORMS_TRAIN(stdev):
    """Returns the transforms required during training"""
    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            # OtsuBinarize(),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )


def TRANSFORMS_EVAL(stdev):
    """Returns the transforms required during evaluation (testing/validation)"""
    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            # OtsuRemoveNoise(),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )
