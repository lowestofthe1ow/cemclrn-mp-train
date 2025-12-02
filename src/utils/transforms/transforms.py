import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as F

from PIL import Image
from torchvision.transforms import InterpolationMode

from src.utils.transforms.OtsuRemoveNoise import OtsuRemoveNoise
from src.utils.transforms.CropToBoundingBox import CropToBoundingBox

TRANSFORMS_PRE = transforms.Compose(
    [
        transforms.RandomInvert(p=1.0),
        # TODO: Look into this, gamma of 2.5 is guesswork
        transforms.Lambda(lambda img: F.adjust_gamma(img, 2.5)),
        transforms.ToTensor(),
        # transforms.GaussianBlur(5),
        OtsuRemoveNoise(),
        CropToBoundingBox(),
        transforms.Resize([155, 220], interpolation=InterpolationMode.NEAREST),
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
            transforms.RandomInvert(p=1.0),
            # TODO: Look into this, gamma of 2.5 is guesswork
            transforms.Lambda(lambda img: F.adjust_gamma(img, 2.5)),
            # Current best doesn't even have this lol
            transforms.RandomAffine(
                degrees=10,
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            # transforms.GaussianBlur(5),
            OtsuRemoveNoise(),
            CropToBoundingBox(),
            transforms.Resize([155, 220], interpolation=InterpolationMode.NEAREST),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )


def TRANSFORMS_EVAL(stdev):
    """Returns the transforms required during evaluation (testing/validation)"""
    return transforms.Compose(
        [
            transforms.RandomInvert(p=1.0),
            # TODO: Look into this, gamma of 2.5 is guesswork
            transforms.Lambda(lambda img: F.adjust_gamma(img, 2.5)),
            transforms.ToTensor(),
            # transforms.GaussianBlur(5),
            OtsuRemoveNoise(),
            CropToBoundingBox(),
            transforms.Resize([155, 220], interpolation=InterpolationMode.NEAREST),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )
