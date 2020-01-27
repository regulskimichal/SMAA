from __future__ import print_function

import os

import cv2
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

from src.commons.resize_images import resize
from src.commons.segmentator import Segmentator, SegmentationError

DATASET_PATH = ".." + os.sep + ".." + os.sep + "dataset"
CHARS_PATH = ".." + os.sep + ".." + os.sep + "characters_test"
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(LABELS)


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
                    chars = find_chars(image, training=True)

                    for i in range(len(chars)):
                        char = chars[i]
                        label = lp[i]
                        dir_path = CHARS_PATH + os.sep + label

                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)

                        count = counts.get(label, 1)
                        path = dir_path + os.sep + str(count).zfill(6) + '.png'
                        cv2.imwrite(path, resize(char))

                        counts[label] = count + 1

            except SegmentationError:
                pass

            except:
                pass


def find_chars(image, training=False):
    segmentator = Segmentator(image)
    return segmentator.detect(training)


if __name__ == '__main__':
    main()
