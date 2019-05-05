import os
import cv2
import numpy as np


def pad_image(img):
    """
    Pad image with 0s to make it square
    :param image: HxWx3 numpy array
    :return: AxAx3 numpy array (square image)
    """
    height, width, _ = img.shape

    if width < height:
        border_width = (height - width) // 2
        padded = cv2.copyMakeBorder(img, 0, 0, border_width, border_width,
                                    cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        padded = cv2.copyMakeBorder(img, border_width, border_width, 0, 0,
                                    cv2.BORDER_CONSTANT, value=0)

    return padded