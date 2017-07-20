from time import time

class LabelAnnealer(object):
    """Keeps track of how many labels we want to collect"""

    def __init__(self, agent_logger, final_timesteps, final_labels, pretrain_labels):
        self._agent_logger = agent_logger
        self._final_timesteps = final_timesteps
        self._final_labels = final_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        """Return the number of labels desired at this point in training. """
        exp_decay_frac = 0.01 ** (self._agent_logger._timesteps_elapsed / self._final_timesteps)  # Decay from 1 to 0
        pretrain_frac = self._pretrain_labels / self._final_labels
        desired_frac = pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay_frac)  # Start with 0.25 and anneal to 0.99
        return desired_frac * self._final_labels

class ConstantLabelSchedule(object):
    def __init__(self, pretrain_labels, seconds_between_labels=3.0):
        self._started_at = None  # Don't initialize until we call n_desired_labels
        self._seconds_between_labels = seconds_between_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        if self._started_at is None:
            self._started_at = time()
        return self._pretrain_labels + (time() - self._started_at) / self._seconds_between_labels
