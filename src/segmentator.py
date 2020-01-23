from collections import namedtuple

import cv2
import imutils
import numpy as np
from skimage import measure
from skimage import segmentation
from skimage.filters import threshold_local

LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])


class SegmentationError(Exception):
    pass


class Segmentator:

    def __init__(self, image, num_chars=7, min_char_w=7):
        self.image = image
        self.num_chars = num_chars
        self.min_char_w = min_char_w

    def detect(self):
        lp = self.detect_character_candidates()
        if lp.success:
            return self.scissor(lp)
        else:
            raise SegmentationError

    def detect_character_candidates(self):
        plate = imutils.resize(self.image, width=100, height=32)

        v = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        t = threshold_local(v, 9, method="gaussian")
        thresh = (v > t).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)

        labels = measure.label(thresh, neighbors=8, background=0)
        char_candidates = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            label_mask = np.zeros(thresh.shape, dtype="uint8")
            label_mask[labels == label] = 255
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                box_x, box_y, box_w, box_h = cv2.boundingRect(c)

                aspect_ratio = box_w / float(box_h)
                solidity = cv2.contourArea(c) / float(box_w * box_h)
                height_ratio = box_h / float(plate.shape[0])

                keep_aspect_ratio = aspect_ratio < 1.0
                keep_solidity = solidity > 0.15
                keep_height = 0.30 < height_ratio < 0.95

                if keep_aspect_ratio and keep_solidity and keep_height:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(char_candidates, [hull], -1, 255, -1)

        char_candidates = segmentation.clear_border(char_candidates)
        contours, _ = cv2.findContours(char_candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > self.num_chars:
            char_candidates, contours = self.prune_candidates(char_candidates, contours)

        return LicensePlate(success=len(contours) == self.num_chars, plate=plate, thresh=thresh,
                            candidates=char_candidates)

    def prune_candidates(self, char_candidates, contours):
        pruned_candidates = np.zeros(char_candidates.shape, dtype="uint8")
        dims = []

        for c in contours:
            box_x, box_y, box_w, box_h = cv2.boundingRect(c)
            dims.append(box_y + box_h)

        dims = np.array(dims)
        diffs = []
        selected = []

        for i in range(0, len(dims)):
            diffs.append(np.absolute(dims - dims[i]).sum())

        for i in np.argsort(diffs)[:self.num_chars]:
            cv2.drawContours(pruned_candidates, [contours[i]], -1, 255, -1)
            selected.append(contours[i])

        return pruned_candidates, selected

    def scissor(self, lp):
        contours, _ = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        chars = []

        for contour in contours:
            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
            d_x = min(self.min_char_w, self.min_char_w - box_w) // 2
            box_x -= d_x
            box_w += d_x * 2

            boxes.append((box_x, box_y, box_x + box_w, box_y + box_h))

        boxes = sorted(boxes, key=lambda b: b[0])

        for (start_x, start_y, end_x, end_y) in boxes:
            chars.append(lp.thresh[start_y: end_y, start_x: end_x])

        return chars

    @staticmethod
    def preprocess_char(char):
        contours, _ = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        char = char[y: y + h, x: x + w]

        return char
