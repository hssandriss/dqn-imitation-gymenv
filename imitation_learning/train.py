from __future__ import print_function
from tensorboard_evaluation import Evaluation
from agent.bc_agent import BCAgent
from utils import *
import matplotlib.pyplot as plt
import gzip
import os
import numpy as np
import pickle
import torch
import sys
sys.path.append("../")


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    X_train = rgb2gray(X_train).reshape(96, 96)/255.0
    X_valid = rgb2gray(X_valid).reshape(96, 96)/255.0
    y_train = np.apply_along_axis(action_to_id, 1, y_train)
    y_valid = np.apply_along_axis(action_to_id, 1, y_valid)
    data_balance(X_train, y_train, 0.5, 0)
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    if history_length > 1:
        # History Xtrain
        X_train_hist = []
        for i in range(X_train.shape[0]):
            state = []
            if (i < history_length):
                for _ in range(history_length-i):
                    state.append(X_train[0])
                for j in range(i):
                    state.append(X_train[j])
            else:
                for j in range(history_length):
                    state.append(X_train[i-j])
            state = np.stack(state, axis=2)
            X_train_hist.append(np.array(state))
        X_train = np.stack(X_train_hist)
        # History Xvalid
        X_valid_hist = []
        for i in range(X_valid.shape[0]):
            state = []
            if (i < history_length):
                for _ in range(history_length-i):
                    state.append(X_valid[0])
                for j in range(i):
                    state.append(X_valid[j])
            else:
                for j in range(history_length):
                    state.append(X_valid[i-j])
            state = np.stack(state, axis=2)
            X_valid_hist.append(np.array(state))
        X_valid = np.stack(X_valid_hist)

    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, history_length, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")
    tensorboard_dir = "./tensorboard_imitation"
    # TODO: specify your agent with the neural network in agents/bc_agent.py
    # agent = BCAgent(...)
    agent = BCAgent(history_length)
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), "train", [
        "train_loss", "train_accuracy", "validation_accuracy"])

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop

    # print(X_train.shape, y_train.shape)
    # print(X_valid.shape, y_valid.shape)
    if history_length > 1:
        X_valid = torch.FloatTensor(X_valid).permute(0, 3, 1, 2).cuda()
        y_valid = torch.LongTensor(y_valid).cuda()
    else:
        X_valid = torch.FloatTensor(X_valid).unsqueeze(1).cuda()
        y_valid = torch.LongTensor(y_valid).cuda()
    for i in range(n_minibatches):
        minibatch_indices = np.random.choice(X_train.shape[0], batch_size)
        y = y_train[minibatch_indices]
        X = X_train[minibatch_indices]
        if history_length > 1:
            X = torch.FloatTensor(X).permute(0, 3, 1, 2).cuda()
            y = torch.LongTensor(y).cuda()
        else:
            X = torch.FloatTensor(X).unsqueeze(1).cuda()
            y = torch.LongTensor(y).cuda()
        # print(X.shape)
        train_loss = agent.update(X, y)
        if i % 10 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            y_pred = agent.predict(X)
            _, act_pred = torch.max(y_pred.data, 1)
            train_acc = (act_pred == y).sum().item()/y_pred.shape[0]
            # val acc
            y_pred = agent.predict(X_valid)
            _, act_pred = torch.max(y_pred.data, 1)
            val_acc = (act_pred == y_valid).sum().item()/y_pred.shape[0]
            print("Iter %i loss %.3f train_acc %.3f val_acc %.3f" % (i, train_loss, train_acc, val_acc))
            tensorboard.write_episode_data(i, eval_dict={"train_loss": train_loss,
                                                         "train_accuracy": train_acc,
                                                         "validation_accuracy": val_acc})
            agent.save(os.path.join(model_dir, "agent_%i.pt" % (i)))
    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    history_length = 5
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length)
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, history_length, n_minibatches=1000, batch_size=64, lr=1e-4)
