from __future__ import print_function
from utils import *
from agent.bc_agent import BCAgent
import json
import os
import gym
import numpy as np
from datetime import datetime
import torch

import sys
sys.path.append("../")


def run_episode(env, agent, history_size=1, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0
    # fix bug of curropted states without rendering in racingcar gym environment
    # env.viewer.window.dispatch_events()
    state = env.reset()
    state = rgb2gray(state).reshape(96, 96) / 255.0
    if history_size > 1:
        state = np.stack([state]*history_length, axis=2)
    while True:
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...
        state_input = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        print(state_input.shape)
        a = agent.predict(state_input)
        _, a = torch.max(a, 1)
        a = id_to_action(a.item())
        next_state, r, done, info = env.step(a)
        episode_reward += r
        next_state = rgb2gray(next_state).reshape(96, 96, 1) / 255.0
        next_state = np.concatenate((state[:, :, 1:], next_state), axis=2)
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True
    path_model = "models/agent_990.pt"
    history_length = 5
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    # agent = BCAgent(...)
    agent = BCAgent(history_length)
    agent.load(path_model)

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_size=history_length, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
