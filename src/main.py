import os

from matplotlib import image
from pandas import read_csv

path = 'D:\\dataset'


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
