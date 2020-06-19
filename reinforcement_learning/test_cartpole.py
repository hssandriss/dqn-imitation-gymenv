import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import MLP
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    # TODO: load DQN agent
    # ...
    states_dim = 4
    action_dim = 1

    Q = MLP(states_dim, action_dim)
    Q_target = MLP(states_dim, action_dim)
    agent = DQNAgent(Q, Q_target, action_dim, double=False)
    agent.load("./../models_cartpole/dqn_agent.pt")
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
