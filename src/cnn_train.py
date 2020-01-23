import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from src.network import Network

CHARS_PATH = ".." + os.sep + "characters"
CHARS_TEST_PATH = ".." + os.sep + "characters_test"
BATCH_SIZE = 100
GAMMA = 0.7
EPOCHS = 14
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=CHARS_PATH,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])])
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=CHARS_TEST_PATH,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5])])
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    model = Network().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
