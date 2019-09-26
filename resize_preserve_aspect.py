""" Data pre-processing functions """

import math

import cv2
import numpy as np
from torchvision import transforms


class ResizeAspectPreserving:  # pylint: disable=R0903
    """ Resizes an image preserving its aspect ratio"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # Warning: PIL returns width, height, and not the opposite!
        width, height = img.size[0:2]
        factor = min(self.size[0] / height, self.size[1] / width)

        new_height = int(math.floor(height * factor + 0.5))  # round
        new_width = int(math.floor(width * factor + 0.5))

        im2 = transforms.functional.resize(img, (new_height, new_width))

        correct_size = transforms.functional.center_crop(im2, self.size)

        return correct_size