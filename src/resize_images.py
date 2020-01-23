import os

import cv2
from imutils import paths

IMAGE_ROOT_PATH = ".." + os.sep + "characters"

if __name__ == '__main__':
    desired_size = 16
    image_paths = list(paths.list_images(IMAGE_ROOT_PATH))

    for image_path in image_paths:
        im = cv2.imread(image_path)
        new_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, new_im)
