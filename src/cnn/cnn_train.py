#!/usr/bin/env python3
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.adadelta import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.cnn.network import Network

CHARS_PATH = ".." + os.sep + ".." + os.sep + "characters"
CHARS_TEST_PATH = ".." + os.sep + ".." + os.sep + "characters_test"
BATCH_SIZE = 1000
GAMMA = 0.7
EPOCHS = 4
LEARNING_RATE = 1


def is_valid_file(file):
    return os.path.isfile(file) and str(file).endswith(".png")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(
            root=CHARS_PATH,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])])
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(
            root=CHARS_TEST_PATH,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])])
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

    model = Network().to(device)
    optimizer = Adadelta(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), 'model.pt')


if __name__ == '__main__':
    main()
