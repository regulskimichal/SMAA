#!/usr/bin/env python3
import os

import cv2
import torch
from torch.utils.data import DataLoader

from src.cnn.network import Network
from src.commons.find_characters import find_chars, LABELS
from src.commons.resize_images import resize

IMAGE_PATH = '..' + os.sep + '..' + os.sep + 'dataset' + os.sep + 's01_l01' + os.sep + '7_25.png'


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
            resized_chars.append(resize(char))

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
                lp += LABELS[pred[0, 0]]

        print(lp)
    except Exception as e:
        print(str(e))
        print("Couldn't find characters in license plate")


if __name__ == '__main__':
    main()
