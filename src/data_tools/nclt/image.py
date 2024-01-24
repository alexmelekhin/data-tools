"""Image processing methods for the NCLT dataset.
"""
import re
from os import PathLike

import cv2
import numpy as np


class Undistort:
    """The class to undistort images using the camera calibration.

    Source: https://robots.engin.umich.edu/nclt/
    """

    def __init__(self, undistort_map_filepath: str | PathLike) -> None:
        """Initialize the undistorter.

        Args:
            undistort_map_filepath: The path to the undistortion (U2D) map file.
        """
        with open(undistort_map_filepath, "r") as f:
            header = f.readline().rstrip()
            chunks = re.sub(r"[^0-9,]", "", header).split(",")
            self.mapu = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(" ")
                self.mapu[int(chunks[0]), int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]), int(chunks[1])] = float(chunks[2])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Undistort the image using the camera calibration with OpenCV.

        Args:
            img: The OpenCV image to undistort.
        """
        return cv2.remap(img, self.mapu, self.mapv, cv2.INTER_CUBIC)


def center_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Center crop the image.

    Args:
        img: The OpenCV image to center crop.
        crop_size: The size of the crop (W, H).

    Raises:
        ValueError: If the image is smaller than the crop size.
    """
    h, w = img.shape[:2]
    if h < crop_size[1] or w < crop_size[0]:
        raise ValueError("Given image is smaller than crop_size")
    center_y, center_x = h // 2, w // 2
    left = center_x - crop_size[0] // 2
    top = center_y - crop_size[1] // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    cropped_img = img[top:bottom, left:right]
    return cropped_img
