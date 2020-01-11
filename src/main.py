import os
import matplotlib.pyplot as plt
import cv2

from src.segmentation import segment_license_plate, segment_characters
from matplotlib import image
from pandas import read_csv
from skimage import img_as_ubyte



path = 'C:\\dataset'


def load():
    loaded_images = {}
    for filename in os.listdir(path):
        if os.path.isdir(path + os.path.sep + filename):
            for image_filename in os.listdir(path + os.path.sep + filename):
                image_path = path + os.path.sep + filename + os.path.sep + image_filename
                loaded_images[filename, image_filename] = image.imread(image_path)

    loaded_table = read_csv(path + os.path.sep + 'trainVal.csv')
    return loaded_images, loaded_table


if __name__ == '__main__':
    images, tables = load()

    for image in images:
        img = img_as_ubyte(images[image])
        plate = segment_license_plate(img)
        plt.imshow(plate)

        chars = segment_characters(plate)
        print(str(image) + ": " + str(len(chars)))





