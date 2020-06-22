import torch
from agent.networks import CNN


class BCAgent:

    def __init__(self, history_size, n_actions=5, lr=0.0004):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.history_size = history_size
        self.num_actions = n_actions
        self.net = CNN(self.history_size, n_actions).cuda()
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        self.net.train()
        y_predicted = self.net(X_batch)
        self.optimizer.zero_grad()
        loss = self.criterion(y_predicted, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        # TODO: forward pass
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(X)
        return outputs

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
