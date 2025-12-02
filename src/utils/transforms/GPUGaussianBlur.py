import torch
import numpy as np
import cv2

from PIL import Image
from torchvision.transforms import v2


class GPUGaussianBlur:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, input_tensor):
        gaussian_blur_transform = v2.GaussianBlur(kernel_size=self.kernel)
        return gaussian_blur_transform(input_tensor.cuda())
