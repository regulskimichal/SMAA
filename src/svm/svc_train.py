from __future__ import print_function

import glob
import os
import pickle
import random

import cv2
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC

from src.commons.find_characters import LABEL_ENCODER, LABELS

SAMPLES = '..' + os.sep + '..' + os.sep + 'characters'


def main():
    data = []
    target = []

    print("Dataset loading")
    for sample_path in sorted(glob.glob(SAMPLES + os.sep + '*')):
        sample_name = sample_path[sample_path.rfind(os.sep) + 1:]
        image_paths = list(paths.list_images(sample_path))
        image_paths = random.sample(image_paths, min(2000, len(image_paths)))

        for image_path in image_paths:
            char = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            data.append(char)
            target.append(sample_name)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=True)
    X_train = (np.array(X_train) / 255.).reshape((len(X_train), -1))
    X_test = (np.array(X_test) / 255.).reshape((len(X_test), -1))
    y_train = LABEL_ENCODER.transform(y_train)
    y_test = LABEL_ENCODER.transform(y_test)

    parameters = {
        "class_weight": ["balanced"],
        "C": [1e1, 1e2, 1e3, 1e4],
        "gamma": [1e-2, 1.8e-1, 1.5e-1, 1e-1]
    }
    print("Training")
    grid_search = GridSearchCV(SVC(), parameters, n_jobs=-1, cv=2)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=LABELS)
    print(report)

    print("Dumping")
    f = open('model_svc_newer.p', "wb")
    f.write(pickle.dumps(grid_search))
    f.close()


if __name__ == '__main__':
    main()
