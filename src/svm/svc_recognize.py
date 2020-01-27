from __future__ import print_function

import os
import pickle

import cv2

from src.commons.find_characters import find_chars, LABEL_ENCODER
from src.commons.resize_images import resize

IMAGE_PATH = '..' + os.sep + '..' + os.sep + 'dataset' + os.sep + 's01_l01' + os.sep + '10_1.png'


def main():
    image = cv2.imread(IMAGE_PATH)
    model = pickle.loads(open('model_svc.p', "rb").read())
    print(model.best_params_)
    print(model.cv_results_)

    try:
        chars = find_chars(image)
        lp = ''
        for char in chars:
            flatten_char = resize(char).reshape(1, -1) / 255.
            lp += LABEL_ENCODER.inverse_transform(model.predict(flatten_char))[0]
            print(lp)
    except Exception as e:
        print(str(e))
        print("Couldn't find characters in license plate")


if __name__ == '__main__':
    main()
