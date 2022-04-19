import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=(20, 20), stride=(10, 10))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=(3, 3))
        self.fc1 = nn.Linear(in_features=6800, out_features=2400)
        self.fc2 = nn.Linear(in_features=2400, out_features=800)
        self.fc3 = nn.Linear(in_features=800, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
