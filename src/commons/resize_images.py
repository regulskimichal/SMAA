import os

import cv2
import numpy as np
from imutils import paths

IMAGE_ROOT_PATH = ".." + os.sep + "characters"
DESIRED_SIZE = 16


def resize(image, size=DESIRED_SIZE):
    startx = (size - image.shape[1]) // 2
    starty = (size - image.shape[0]) // 2
    result = np.zeros((16, 16))
    result[starty:starty + image.shape[0], startx:startx + image.shape[1]] = image

    return result


if __name__ == '__main__':
    image_paths = list(paths.list_images(IMAGE_ROOT_PATH))

    for image_path in image_paths:
        im = cv2.imread(image_path)
        im = resize(im)
        cv2.imwrite(image_path, im)
