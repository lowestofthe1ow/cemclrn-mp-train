import torch
import numpy as np
import cv2
from PIL import Image


class OtsuRemoveNoise:
    """Removes noise from a grayscale image to produce a "black-and-white" image"""

    def __call__(self, input_tensor):
        img_np = (input_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Perform Otsu's method for finding a threshold
        # All pixels < threshold get set to 0
        _, binary_img_np = cv2.threshold(
            img_np, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU
        )

        binary_tensor = torch.from_numpy(binary_img_np).float().unsqueeze(0) / 255.0

        return binary_tensor
