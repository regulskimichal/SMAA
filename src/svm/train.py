from __future__ import print_function

import glob
import os
import pickle
import random

import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SAMPLES = '..' + os.sep + '..' + os.sep + 'characters'


def main():
    data = []
    target = []

    print("Dataset loading")
    for sample_path in sorted(glob.glob(SAMPLES + os.sep + '*')):
        sample_name = sample_path[sample_path.rfind(os.sep) + 1:]
        image_paths = list(paths.list_images(sample_path))
        image_paths = random.sample(image_paths, min(1000, len(image_paths)))

        for image_path in image_paths:
            char = cv2.imread(image_path)
            data.append(cv2.cvtColor(char, cv2.COLOR_BGR2GRAY))
            target.append(sample_name)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=True)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Training")
    char_model = SVC(gamma=0.001, max_iter=1000)
    char_model.fit(X_train.reshape((len(X_train), -1)), y_train)

    print("Dumping")
    f = open('model.p', "wb")
    f.write(pickle.dumps(char_model))
    f.close()


if __name__ == '__main__':
    main()
