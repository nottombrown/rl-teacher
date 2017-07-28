import gym
import numpy as np
from gym import spaces

class Particle(gym.Env):
    def __init__(self):
        """
        Simple particle environment where reward is -distance_from_origin^2 and actions are deltas to the particle location.
        Useful for visualizing a learned rewarder
        """
        box = spaces.Box(float('-inf'), float('inf'), shape=(2,))
        self.action_space = box  # (particle_x_delta, particle_y_delta)
        self.observation_space = box  # (particle_x, particle_y)

    def _reset(self):
        self.loc = np.random.randn(2)
        return np.copy(self.loc)

    def _step(self, action):
        self.loc += action
        reward = -np.sum(np.square(self.loc))
        return np.copy(self.loc), reward, False, {}

class ParticleRewardPredictor(object):
    def predict_reward(self, path):
        # N.B. Observations include the original reset obs, and do not contain the obs resulting from the final action
        obs = path["obs"]
        action = path["actions"]

        loc = obs + action
        return -np.sum(np.square(loc), axis=tuple(i for i in range(1, obs.ndim)))
