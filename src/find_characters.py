from __future__ import print_function

import os

import cv2
from pandas import read_csv

from src.segmentator import Segmentator, SegmentationError

DATASET_PATH = ".." + os.sep + "dataset"
CHARS_PATH = ".." + os.sep + "characters_test"


def main():
    lp_descriptions = read_csv(DATASET_PATH + os.path.sep + 'trainVal.csv')
    counts = {}

    for _, lp_description in lp_descriptions.iterrows():
        if lp_description['train'] == 0:
            try:
                image_path = DATASET_PATH + os.sep + lp_description['image_path'].replace('/', os.sep)
                if os.path.exists(image_path):
                    lp = lp_description['lp']
                    image = cv2.imread(image_path)
                    segmentator = Segmentator(image, num_chars=len(lp), min_char_w=8)
                    chars = segmentator.detect()

                    for index in range(len(chars)):
                        label = lp[index]
                        dir_path = CHARS_PATH + os.sep + label

                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)

                        count = counts.get(label, 1)
                        path = dir_path + os.sep + str(count).zfill(6) + '.png'
                        cv2.imwrite(path, chars[index])

                        counts[label] = count + 1

            except SegmentationError:
                pass

            except:
                pass


if __name__ == '__main__':
    main()
