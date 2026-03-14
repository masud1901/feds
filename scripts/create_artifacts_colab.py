"""Create initial_global_model_MNIST and K pickle for Colab/local runs."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
os.chdir(REPO)

p = __import__("pickle")
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
    if not os.path.exists("initial_global_model_MNIST"):
        net = Net()
        params = [val.cpu().numpy() for _, val in net.state_dict().items()]
        initial_model = [(params, 1)]
        with open("initial_global_model_MNIST", "wb") as f:
            p.dump(initial_model, f)
        print("Created initial_global_model_MNIST")
    else:
        print("initial_global_model_MNIST already exists")

    k_pickle = "K_alpha=10_gamma=10_test=0.pickle"
    if not os.path.exists(k_pickle):
        d = 582026
        k_list = [d // 2] * 10
        with open(k_pickle, "wb") as f:
            p.dump(k_list, f)
        print("Created", k_pickle)
    else:
        print(k_pickle, "already exists")
