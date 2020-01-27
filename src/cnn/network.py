import torch
import torch.nn.functional as F
from torch import nn

from src.commons.find_characters import LABELS
from src.commons.resize_images import DESIRED_SIZE

CONV1_KERNEL_SIZE = 3
CONV2_KERNEL_SIZE = 3

CONV1_OUT_SIZE = 16
CONV2_OUT_SIZE = 64
LINEAR1_OUT_SIZE = 256


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, CONV1_OUT_SIZE, CONV1_KERNEL_SIZE)
        self.conv2 = nn.Conv2d(CONV1_OUT_SIZE, CONV2_OUT_SIZE, CONV2_KERNEL_SIZE)
        self.fc1 = nn.Linear(DESIRED_SIZE * DESIRED_SIZE * CONV1_KERNEL_SIZE * CONV2_KERNEL_SIZE, LINEAR1_OUT_SIZE)
        self.fc2 = nn.Linear(LINEAR1_OUT_SIZE, len(LABELS))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
