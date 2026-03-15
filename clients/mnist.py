# -*- coding: utf-8 -*-
"""
MNIST Federated Learning Client — fixed for Flower >= 1.0
"""

from collections import OrderedDict
import warnings
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common.logger import log
from logging import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import numpy as np
import argparse

warnings.filterwarnings("ignore", category=Warning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model
# =============================================================================
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# Train / Test
# =============================================================================
def train(net, trainloader, epochs, seed):
    """Train the network. Returns average loss."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.train()
    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def test(net, testloader):
    """Validate on the full test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# =============================================================================
# Datasets
# =============================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        length = int(len(dataset) / 10)
        self.idxs = list(np.arange(length * seed, length * seed + length))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetNonIID(Dataset):
    def __init__(self, dataset, seed):
        seed = int(seed)
        targets = [[0, 1], [0, 1], [2, 3], [2, 3], [4, 5],
                   [4, 5], [6, 7], [6, 7], [8, 9], [8, 9]]
        self.userdataset = [(img, label) for img, label in dataset
                            if label in targets[seed]]

    def __len__(self):
        return len(self.userdataset)

    def __getitem__(self, item):
        image, label = self.userdataset[item]
        return image, label


def load_data(seed, IID):
    """Load MNIST training and test sets."""
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = MNIST('../data/mnist/', train=True,  download=True, transform=trans)
    testset  = MNIST('../data/mnist/', train=False, download=True, transform=trans)
    if IID:
        trainloader = DataLoader(DatasetSplit(trainset, seed), batch_size=32, shuffle=True)
        testloader  = DataLoader(DatasetSplit(testset,  seed), batch_size=32)
    else:
        trainloader = DataLoader(DatasetNonIID(trainset, seed), batch_size=32, shuffle=True)
        testloader  = DataLoader(DatasetNonIID(testset,  seed), batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


# =============================================================================
# Flower Client — fixed for Flower >= 1.0
# =============================================================================
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, args, net, trainloader, testloader, num_examples):
        super().__init__()
        self.args = args
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    # FIX 1: config parameter is required in Flower >= 1.0
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss = train(self.net, self.trainloader, 1, self.args.seed)
        return self.get_parameters(config={}), self.num_examples["trainset"], {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


# =============================================================================
# Main
# =============================================================================
def main():
    iid = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducibility and data split")
    args = parser.parse_args()

    log(INFO, f"Using seed {args.seed} for reproducibility")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = Net().to(DEVICE)
    trainloader, testloader, num_examples = load_data(args.seed, IID=iid)

    if iid:
        log(INFO, "Using IID dataset")
    else:
        log(INFO, "Using Non-IID dataset")

    client = MNISTClient(args, net, trainloader, testloader, num_examples)

    # FIX 2: use server_address= keyword arg (required in Flower >= 1.0)
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )


if __name__ == "__main__":
    main()