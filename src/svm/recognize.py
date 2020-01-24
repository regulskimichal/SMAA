from __future__ import print_function

import os
import pickle

import cv2
import numpy as np

from src.commons.find_characters import find_chars
from src.commons.resize_images import resize

IMAGE_PATH = '..' + os.sep + '..' + os.sep + 'dataset' + os.sep + 's01_l01' + os.sep + '11_6.png'
DICT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def main():
    image = cv2.imread(IMAGE_PATH)
    model = pickle.loads(open('model.p', "rb").read())

    try:
        chars = find_chars(image)
        lp = ''
        for char in chars:
            resized_char = resize(cv2.cvtColor(char[:, :, np.newaxis], cv2.COLOR_GRAY2RGB)).reshape(1, -1)
            lp += model.predict(resized_char)[0]

        print(lp)
    except Exception as e:
        print(str(e))
        print("Couldn't find characters in license plate")


if __name__ == '__main__':
    main()
