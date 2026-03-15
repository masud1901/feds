"""Create initial_global_model_CIFAR and K_initial.json for CIFAR-10 runs."""
import os
import sys
import json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
os.chdir(REPO)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
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
    model_path = "initial_global_model_CIFAR.npz"
    if not os.path.exists(model_path):
        net = Net()
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        np.savez_compressed(model_path, *params)
        print("Created", model_path)
    else:
        print(model_path, "already exists")

    k_path = "K_initial.json"
    if not os.path.exists(k_path):
        d = 62006
        k_list = [d // 2] * 10
        with open(k_path, "w") as f:
            json.dump({"k_list": k_list}, f)
        print("Created", k_path)
    else:
        print(k_path, "already exists")
