"""Create initial_global_model_CIFAR for CIFAR-10 experiments."""
import os
import sys
import pickle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
os.chdir(REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """CIFAR-10 CNN model."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    if not os.path.exists("initial_global_model_CIFAR"):
        net = Net()
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        initial_model = [(params, 1)]
        with open("initial_global_model_CIFAR", "wb") as f:
            pickle.dump(initial_model, f)
        print("Created initial_global_model_CIFAR")
        print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
    else:
        print("initial_global_model_CIFAR already exists")

    # Create K pickle
    k_pickle = "K_CIFAR_alpha=10_gamma=10_test=0.pickle"
    if not os.path.exists(k_pickle):
        d = 62006  # CIFAR-10 model parameters
        k_list = [d // 2] * 10
        with open(k_pickle, "wb") as f:
            pickle.dump(k_list, f)
        print(f"Created {k_pickle} with K={d//2}")
    else:
        print(f"{k_pickle} already exists")
