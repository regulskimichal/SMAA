#!/usr/bin/env python3
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cnn.network import Network
from src.commons.find_characters import find_chars
from src.commons.resize_images import resize

IMAGE_PATH = '..' + os.sep + '..' + os.sep + 'dataset' + os.sep + 's01_l01' + os.sep + '10_1.png'
DICT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    image = cv2.imread(IMAGE_PATH)
    model = Network().to(device)
    model.load_state_dict(torch.load('model.pt'))
    try:
        resized_chars = []
        chars = find_chars(image)
        for char in chars:
            resized_chars.append(resize(cv2.cvtColor(char[:, :, np.newaxis], cv2.COLOR_GRAY2RGB)))

        data_loader = DataLoader(torch.utils.data.TensorDataset(torch.tensor(resized_chars).unsqueeze(1)),
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=12,
                                 pin_memory=True)

        model.eval()
        lp = ''
        with torch.no_grad():
            for data in data_loader:
                data = data[0].to(device, dtype=torch.float)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                lp += DICT[pred[0, 0]]

        print(lp)
    except Exception as e:
        print(str(e))
        print("Couldn't find characters in license plate")


if __name__ == '__main__':
    main()
