from copy import deepcopy
import os.path as osp
from collections import deque

import numpy as np
import tensorflow as tf

CLIP_LENGTH = 1.5

def make_summary_writer(name):
    logs_path = osp.expanduser('~/tb/rl-teacher/%s' % (name))
    return tf.summary.FileWriter(logs_path)

def add_simple_summary(summary_writer, tag, simple_value, step):
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)]), step)

def _pad_with_end_state(path, desired_length):
    # Assume path length is at least 1.
    if len(path["obs"]) >= desired_length:
        return path
    path = deepcopy(path)
    for k in path:
        # Casting path[k] as a list is necessary to avoid issues with concatinating numpy arrays.
        path[k] = list(path[k]) + [path[k][-1] for _ in range(desired_length - len(path[k]))]
    return path

class AgentLogger(object):
    """Tracks the performance of an arbitrary agent"""

    def __init__(self, summary_writer, timesteps_per_summary=int(1e3)):
        self.summary_step = 0
        self.timesteps_per_summary = timesteps_per_summary

        self._timesteps_elapsed = 0
        self._timesteps_since_last_training = 0

        n = 100
        self.last_n_paths = deque(maxlen=n)
        self.summary_writer = summary_writer

    def get_recent_paths_with_padding(self):
        """
        Returns the last_n_paths, but with short paths being padded out so the result
        can safely be made into an array.
        """
        if len(self.last_n_paths) == 0:
            return []
        max_len = max([len(path["obs"]) for path in self.last_n_paths])
        return [_pad_with_end_state(path, max_len) for path in self.last_n_paths]

    def log_episode(self, path):
        self._timesteps_elapsed += len(path["obs"])
        self._timesteps_since_last_training += len(path["obs"])
        self.last_n_paths.append(path)

        if self._timesteps_since_last_training >= self.timesteps_per_summary:
            self.summary_step += 1
            if 'new' in path: # PPO puts multiple episodes into one path
                last_n_episode_scores = [np.sum(path["original_rewards"]).astype(float) / np.sum(path["new"])
                    for path in self.last_n_paths]
            else:
                last_n_episode_scores = [np.sum(path["original_rewards"]).astype(float) for path in self.last_n_paths]

            self.log_simple("agent/true_reward_per_episode", np.mean(last_n_episode_scores))
            self.log_simple("agent/total_steps", self._timesteps_elapsed, )
            self._timesteps_since_last_training -= self.timesteps_per_summary
            self.summary_writer.flush()

    def log_simple(self, tag, simple_value, debug=False):
        add_simple_summary(self.summary_writer, tag, simple_value, self.summary_step)
        if debug:
            print("%s    =>    %s" % (tag, simple_value))
