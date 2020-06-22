import numpy as np

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE             # BRAKE: 4
    else:
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


def data_balance(X, y, p, action):
    idx_straight = np.where(y == action)[0]
    idx_others = np.where(y != action)[0]
    mask = np.where(np.random.rand(idx_straight.shape[0]) <= p)[0]
    idx_kept = idx_straight[mask]
    # print(idx_kept.shape, idx_others.shape)
    idx_kept = np.hstack((idx_kept, idx_others))
    return X[idx_kept], y[idx_kept]


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
