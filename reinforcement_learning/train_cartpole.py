import sys
import numpy as np
import gym
import itertools as it
from utils import EpisodeStats
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP


sys.path.append("../")


def run_episode(env, agent, deterministic, do_training=True, rendering=True, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, race=False, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, num_eval_episodes, eval_cycle, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard_train = Evaluation(os.path.join(tensorboard_dir, "train"), "train", [
                                   "episode_reward_train", "episode_length_train", "a_0", "a_1"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "eval"), "eval",
                                  ["episode_reward_valid", "episode_length_valid"])
    # training
    early = 3
    for i in range(num_episodes):
        # if i % 500 == 0 and i != 0:
        print("training episode: ", i)
        stats = run_episode(env, agent, deterministic=False, rendering=True,  do_training=True)
        print(stats.episode_reward)
        tensorboard_train.write_episode_data(i, eval_dict={"episode_reward_train": stats.episode_reward,
                                                           "episode_length_train": len(stats.actions_ids),
                                                           "a_0": stats.get_action_usage(0),
                                                           "a_1": stats.get_action_usage(1)})

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # print(stats.episode_reward)
        if i % eval_cycle == 0:
            cumreward = 0.
            cumlength = 0.
            for j in range(num_eval_episodes):
                # print("evaluation episode", j)
                stats = run_episode(env, agent, deterministic=True, rendering=True, do_training=False)
                cumreward += stats.episode_reward
                cumlength += len(stats.actions_ids)
            tensorboard_eval.write_episode_data(
                i, eval_dict={"episode_reward_valid": cumreward/num_eval_episodes,
                              "episode_length_valid": cumlength/num_eval_episodes})
            if cumreward/num_eval_episodes >= 950:
                print("Saving model ...")
                agent.save(os.path.join(model_dir, "dqn_agent_fixed_decay.pt"))
                early -= 1
                if not early:
                    break

        # # store model.
        # if i % eval_cycle == 0 or i >= (num_episodes - 1):

    tensorboard_train.close_session()
    tensorboard_eval.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 10       # evaluate every 10 episodes
    num_episodes = 10000
    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # ...
    Q_target = MLP(state_dim, num_actions)
    Q = MLP(state_dim, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # agent = DQNAgent(Q, Q_target, num_actions, double=True, history_length=1e6)
    agent = DQNAgent(Q, Q_target, num_actions, double=True, epsilon=0.99, eps_decay=True, history_length=1e6)
    # 3. train DQN agent with train_online(...)
    train_online(env, agent, num_episodes, num_eval_episodes, eval_cycle)
