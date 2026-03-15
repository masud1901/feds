# -*- coding: utf-8 -*-
"""
Speech Commands Federated Learning Client — fixed for Flower >= 1.0
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
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import argparse
import os
import torchaudio






warnings.filterwarnings("ignore", category=Warning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model
# =============================================================================
class Net(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        # self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        # self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        # self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        # self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze()


def train(net, trainloader, epochs, seed):
    """Train the network. Returns average loss."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    new_sample_rate = 8000
    sample_rate = 16000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    net.train()
    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for audio, labels in trainloader:
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            audio = transform(audio)
            optimizer.zero_grad()
            loss = criterion(net(audio), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def test(net, testloader):
    """Validate on the full test set."""
    criterion = torch.nn.CrossEntropyLoss()
    new_sample_rate = 8000
    sample_rate = 16000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for audio, labels in testloader:
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            audio = transform(audio)
            outputs = net(audio)
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
        return self.dataset[self.idxs[item]]


class DatasetNonIID(Dataset):
    def __init__(self, dataset, seed):
        seed = int(seed)
        targets = [
            ["right", "house", "go", "seven", "backward", "down", "bed"], ["right", "house", "go", "seven", "backward", "down", "bed"],
            ["follow", "marvin", "nine", "three", "eight", "left", "cat"], ["follow", "marvin", "nine", "three", "eight", "left", "cat"],
            ["happy", "visual", "zero", "stop", "four", "tree", "wow"], ["happy", "visual", "zero", "stop", "four", "tree", "wow"],
            ["off", "up", "six", "two", "forward", "learn", "five"], ["off", "up", "six", "two", "forward", "learn", "five"],
            ["sheila", "bird", "yes", "dog", "no", "on", "one"], ["sheila", "bird", "yes", "dog", "no", "on", "one"],
        ]
        self.userdataset = [(w, t, label) for w, t, label, *_ in dataset if label in targets[seed]]

    def __len__(self):
        return len(self.userdataset)

    def __getitem__(self, item):
        return self.userdataset[item]


def load_data(seed, IID):
    """Load Speech Commands training and test sets."""

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("./", download=True)
    
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]
    
    
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    def label_to_index(word):
        return torch.tensor(labels.index(word))

    def index_to_label(index):
        return labels[index]

    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
    
    
    def collate_fn(batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [label_to_index(label)]
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)
        return tensors, targets

    if IID:
        trainloader = DataLoader(DatasetSplit(train_set, seed), batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        testloader = DataLoader(DatasetSplit(test_set, seed), batch_size=32, collate_fn=collate_fn, pin_memory=True)
    else:
        trainloader = DataLoader(DatasetNonIID(train_set, seed), batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        testloader = DataLoader(DatasetNonIID(test_set, seed), batch_size=32, collate_fn=collate_fn, pin_memory=True)
    num_examples = {"trainset": len(train_set), "testset": len(test_set)}
    return trainloader, testloader, num_examples




# =============================================================================
# Flower Client — fixed for Flower >= 1.0
# =============================================================================
class SpeechCommandsClient(fl.client.NumPyClient):
    def __init__(self, args, net, trainloader, testloader, num_examples):
        super().__init__()
        self.args = args
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss = train(self.net, self.trainloader, 5, self.args.seed)
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

    net = Net(n_input=1, n_output=35).to(DEVICE)
    trainloader, testloader, num_examples = load_data(args.seed, IID=iid)

    if iid:
        log(INFO, "Using IID dataset")
    else:
        log(INFO, "Using Non-IID dataset")

    client = SpeechCommandsClient(args, net, trainloader, testloader, num_examples)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )


if __name__ == "__main__":
    main()