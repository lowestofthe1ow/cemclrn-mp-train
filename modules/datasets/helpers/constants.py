import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2

from torchvision.transforms import InterpolationMode

TRANSFORMS_PRE = transforms.Compose(
    [
        transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
        transforms.RandomInvert(p=1.0),
    ]
)


def TRANSFORMS_TRAIN(stdev):
    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.PILToTensor(),
            transforms_v2.ToDtype(torch.float32, scale=False),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )
