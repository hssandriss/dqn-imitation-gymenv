import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3):
        # TODO : define layers of a convolutional neural network
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(history_length)
        self.conv1 = nn.Conv2d(history_length, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dense = nn.Linear(64*8*8, 512)
        self.out = nn.Linear(512, n_classes)

    def forward(self, x):
        # TODO: compute forward pass
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.contiguous()
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = self.relu(x)
        x = self.out(x)
        return x
