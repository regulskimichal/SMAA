import os

from matplotlib import image
from pandas import read_csv


if __name__ == '__main__':
    path = '/home/mregulski/dataset'
    loaded_images = {}
    for filename in os.listdir(path):
        if os.path.isdir(path + '/' + filename) and filename == 's01_l01':
            for image_filename in os.listdir(path + '/' + filename):
                loaded_images[filename + '/' + image_filename] = image.imread(path + '/' + filename + '/' + image_filename)
    table = read_csv(path + '/trainVal.csv')
    print(loaded_images['s01_l01/1_1.png'])
    print(len(loaded_images))
    print(table)
