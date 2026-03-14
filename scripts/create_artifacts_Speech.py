"""Create initial_global_model_Speech for Speech Commands experiments."""
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
    """Speech Commands 1D CNN model."""
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
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


if __name__ == "__main__":
    if not os.path.exists("initial_global_model_Speech"):
        net = Net(n_input=1, n_output=35)
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        initial_model = [(params, 1)]
        with open("initial_global_model_Speech", "wb") as f:
            pickle.dump(initial_model, f)
        print("Created initial_global_model_Speech")
        print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")
    else:
        print("initial_global_model_Speech already exists")

    # Create K pickle
    k_pickle = "K_Speech_alpha=10_gamma=10_test=0.pickle"
    if not os.path.exists(k_pickle):
        d = 35000  # Approximate Speech model parameters
        k_list = [d // 2] * 10
        with open(k_pickle, "wb") as f:
            pickle.dump(k_list, f)
        print(f"Created {k_pickle} with K={d//2}")
    else:
        print(f"{k_pickle} already exists")
