import torch
import numpy as np
import cv2
from PIL import Image


class CropToBoundingBox:
    """Removes noise from a grayscale image to produce a black-and-white image"""

    def __call__(self, input_tensor):
        img_np = input_tensor.squeeze(0).cpu().numpy()

        content = img_np == 1.0
        y_coords, x_coords = np.where(content)

        if y_coords.size == 0:
            return torch.from_numpy(img_np).float().unsqueeze(0)

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        cropped_img_np = img_np[y_min : y_max + 1, x_min : x_max + 1]

        return torch.from_numpy(cropped_img_np).float().unsqueeze(0)
