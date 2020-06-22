import numpy as np
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import tensorflow.compat.v1 as tf
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, double=False, gamma=0.95, batch_size=64, epsilon=0.9, tau=0.01, lr=1e-4, history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.double = double
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # 2. sample next batch and perform batch update:
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(
            self.batch_size)
        batch_states = torch.FloatTensor(batch_states).cuda()
        batch_actions = torch.FloatTensor(batch_actions).unsqueeze(1).cuda()
        batch_next_states = torch.FloatTensor(batch_next_states).cuda()
        batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).cuda()
        batch_dones = torch.FloatTensor(batch_dones).unsqueeze(1).cuda()

        #      2.1 compute td targets and loss
        #             td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        # print(batch_next_states.shape, batch_states.shape)
        if not self.double:
            # DQN
            self.Q_target.eval()
            with torch.no_grad():
                target_state_action_values = self.Q_target(batch_next_states).detach()
                td_target = batch_rewards + (1-batch_dones) * self.gamma * \
                    torch.max(target_state_action_values, 1)[0].unsqueeze(1)

            self.Q.train()
            state_action_values = self.Q(batch_states).gather(1, batch_actions.long())
            self.optimizer.zero_grad()
            loss = self.loss_function(td_target, state_action_values)
            #      2.2 update the Q network
            loss.backward()
            for param in self.Q.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        else:
            # DDQN
            self.Q_target.eval()
            with torch.no_grad():
                model_state_action_values = self.Q(batch_next_states)
                target_actions = torch.max(model_state_action_values, 1)[1].unsqueeze(1)
                batch_target_values = self.Q_target(batch_next_states).gather(1, target_actions.long()).detach()
                td_target = batch_rewards + (1-batch_dones) * self.gamma * batch_target_values

            self.Q.train()
            state_action_values = self.Q(batch_states).gather(1, batch_actions.long())
            self.optimizer.zero_grad()
            loss = self.loss_function(td_target, state_action_values)
            #      2.2 update the Q network
            loss.backward()
            for param in self.Q.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        for name, param in self.Q.named_parameters():
            if param.grad is None:
                print('None ', name, param.grad)
                raise Exception("Gradients are not getting computed")
        #      2.3 call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic, race):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        # Epsilon decay
        self.epsilon = max(self.epsilon * 0.995, 0.05)
        r = np.random.uniform()
        # self.Q.eval()
        state = torch.Tensor(state).unsqueeze(0).cuda()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            with torch.no_grad():
                action_id = torch.argmax(self.Q(state), dim=1).cpu().item()
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
            if race:
                action_id = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.1, 0.1, 0.5, 0.1])
            else:
                action_id = np.random.randint(0, self.num_actions)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
