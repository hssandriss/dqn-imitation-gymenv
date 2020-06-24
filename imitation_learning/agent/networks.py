import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=5):
        super(CNN, self).__init__()
        self.batchnorm0 = nn.BatchNorm2d(history_length)
        self.conv1 = nn.Conv2d(history_length, 6, 7)  # batch * 6 * 90 *90
        self.pool1 = nn.MaxPool2d(5, 5)  # batch * 6 * 18 *18
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3)  # batch * 16 * 16 * 16
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # batch * 16 * 8 * 8 # or 4 and becomes 16*4*4
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.drouput = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, n_classes)

        # TODO : define layers of a convolutional neural network

    def forward(self, x):
        # TODO: compute forward pass
        x = self.batchnorm0(x)
        x = self.batchnorm1(self.conv1(x))
        x = self.pool1(F.relu(x))  # batch before relu !
        x = self.batchnorm2(self.conv2(x))
        x = self.pool2(F.relu(x))
        x = x.contiguous()
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.drouput(x)
        x = self.fc2(x)  # check output
        return x


# class CNN(nn.Module):

#     def __init__(self, history_length=0, n_classes=3):
#         # TODO : define layers of a convolutional neural network
#         super(CNN, self).__init__()
#         self.relu = nn.ReLU()
#         self.bn0 = nn.BatchNorm2d(history_length)
#         self.conv1 = nn.Conv2d(history_length, 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.dense = nn.Linear(64*8*8, 512)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.dropout = nn.Dropout(0.3)
#         self.out = nn.Linear(512, n_classes)

#     def forward(self, x):
#         # TODO: compute forward pass
#         x = self.bn0(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = x.contiguous()
#         x = x.view(x.shape[0], -1)
#         x = self.dense(x)
#         x = self.bn4(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.out(x)
#         return x
