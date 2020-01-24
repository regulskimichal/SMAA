import os

import cv2
from imutils import paths

IMAGE_ROOT_PATH = ".." + os.sep + "characters"
DESIRED_SIZE = 16


def resize(image, size=DESIRED_SIZE):
    old_size = image.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    image_paths = list(paths.list_images(IMAGE_ROOT_PATH))

    for image_path in image_paths:
        im = cv2.imread(image_path)
        im = resize(im)
        cv2.imwrite(image_path, im)
