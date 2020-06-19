import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


"""
Car Racing Network
"""


class CNN(nn.Module):

    def __init__(self, num_actions):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(128, num_actions)
        )
        # # Init with cuda if available
        # if torch.cuda.is_available():
        #     self.cuda()
        # self.apply(self.weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
