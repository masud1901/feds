"""Create initial_global_model_MNIST and K_initial.json for Colab/local runs."""
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
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model_path = "initial_global_model_MNIST.npz"
    if not os.path.exists(model_path):
        net = Net()
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        np.savez_compressed(model_path, *params)
        print("Created", model_path)
    else:
        print(model_path, "already exists")

    k_path = "K_initial.json"
    if not os.path.exists(k_path):
        d = 582026
        k_list = [d // 2] * 10
        with open(k_path, "w") as f:
            json.dump({"k_list": k_list}, f)
        print("Created", k_path)
    else:
        print(k_path, "already exists")
