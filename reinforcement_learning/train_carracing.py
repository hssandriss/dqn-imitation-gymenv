# export DISPLAY=:0
from utils import EpisodeStats, rgb2gray, id_to_action
import itertools as it
from tensorboard_evaluation import *
from agent.networks import CNN
from agent.dqn_agent import DQNAgent
import gym
import numpy as np
import sys
sys.path.append("../")

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    state = state.transpose(2, 0, 1)
    while True:
        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state, deterministic=deterministic, race=True)
        action = id_to_action(action_id)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)
        next_state = next_state.transpose(2, 0, 1)
        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), "train_carracing", [
                             "episode_reward", "straight", "left", "right", "accel", "brake"])
    tensorboard_eval = Evaluation(os.path.join(tensorboard_dir, "eval"), "eval_carracing",
                                  ["episode_reward_valid", "episode_length_valid"])
    max_timesteps = 200
    for i in range(num_episodes):
        print("epsiode %d" % i)
        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        stats = run_episode(env, agent, max_timesteps=max_timesteps,
                            history_length=history_length, deterministic=False,
                            skip_frames=2, do_training=True, rendering=True)
        print(stats.episode_reward)
        max_timesteps = min(3000, max_timesteps+10)

        tensorboard.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward,
                                                     "straight": stats.get_action_usage(STRAIGHT),
                                                     "left": stats.get_action_usage(LEFT),
                                                     "right": stats.get_action_usage(RIGHT),
                                                     "accel": stats.get_action_usage(ACCELERATE),
                                                     "brake": stats.get_action_usage(BRAKE)
                                                     })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
            cumreward = 0.
            cumlength = 0.
            for j in range(num_eval_episodes):
                # print("evaluation episode", j)
                stats = run_episode(env, agent, deterministic=True, rendering=True,
                                    do_training=False, history_length=history_length, max_timesteps=max_timesteps)
                cumreward += stats.episode_reward
                cumlength += len(stats.actions_ids)
            tensorboard_eval.write_episode_data(
                i, eval_dict={"episode_reward_valid": cumreward/num_eval_episodes,
                              "episode_length_valid": cumlength/num_eval_episodes})

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "carrace_agent.ckpt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    hist = 3
    num_actions = 5
    env = gym.make('CarRacing-v0').unwrapped

    # TODO: Define Q network, target network and DQN agent
    # ...
    hist = 3
    num_actions = 5
    Q_target = CNN(hist+1, num_actions)
    Q = CNN(hist+1, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(Q, Q_target, num_actions, double=False, history_length=1e6)
    # agent = DQNAgent(Q, Q_target, num_actions, double=False, epsilon = 0.99, eps_decay = True, history_length=1e6)
    # 3. train DQN agent with train_online(...)
    train_online(env, agent, num_episodes=1000, history_length=hist, model_dir="./models_carracing")
