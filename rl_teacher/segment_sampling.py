import numpy as np
import tensorflow as tf
from parallel_trpo.rollouts import ParallelRollout

from rl_teacher.envs import get_timesteps_per_episode, make_with_torque_removed
from rl_teacher.video import write_segment_to_video

CLIP_LENGTH = 1.5

def create_segment_q_states(segment):
    obs_Ds = np.reshape(segment["obs"], (len(segment["obs"]), -1))  # Flatten observation space
    act_Ds = np.reshape(segment["action"], (len(segment["action"]), -1))  # Flatten action space
    return np.concatenate([obs_Ds, act_Ds], axis=1)

def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path['obs'])

    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = {
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', 'action', 'original_rewards', 'human_obs']}

    # Add q_states
    segment['q_states'] = create_segment_q_states(segment)
    return segment

class SegmentVideoRecorder(object):
    def __init__(self, predictor, env, save_dir, n_desired_videos_per_checkpoint=1, checkpoint_interval=10):
        self.predictor = predictor
        self.env = env
        self.n_desired_videos_per_checkpoint = n_desired_videos_per_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir

        self._counter = 0  # Internal counter of how many videos we've saved at a given iteration.

    def path_callback(self, path, iteration):
        if iteration % self.checkpoint_interval == 0:
            if self._counter < self.n_desired_videos_per_checkpoint:
                fname = '%s/run_%s_%s.mp4' % (self.save_dir, iteration, self._counter)
                print("Saving video of run %s_%s to %s" % (iteration, self._counter, fname))
                full_run = sample_segment_from_path(path, len(path['obs']))
                write_segment_to_video(full_run, fname, self.env)
                self._counter += 1
        else:
            self._counter = 0

    def predict_reward(self, path):
        return self.predictor.predict_reward(path)

class RandomRolloutSegmentCollector(object):
    def __init__(self, n_desired_segments, fps):
        self.n_desired_segments = n_desired_segments
        self.fps = fps
        self.segments = []

    def path_callback(self, path, iteration):
        segment = sample_segment_from_path(path, int(CLIP_LENGTH * self.fps))
        if segment:
            self.segments.append(segment)

        if len(self.segments) % 10 == 0 and len(self.segments) > 0:
            print("Collected %s/%s segments" % (len(self.segments), self.n_desired_segments))

        if len(self.segments) >= self.n_desired_segments:
            raise SuccessfullyCollectedSegments()

    def predict_reward(self, path):
        # Reward is unused during random rollout so just return a tiny value
        return np.ones(len(path["obs"])) * 1e-9

class SuccessfullyCollectedSegments(Exception):
    pass

def segments_from_rand_rollout(seed, env_id, env, n_segments, workers=4):
    collector = RandomRolloutSegmentCollector(n_segments, fps=env.fps)
    max_timesteps_per_episode = get_timesteps_per_episode(env)
    timesteps_per_batch = 8000
    try:
        with tf.Graph().as_default():
            rollouts = ParallelRollout(env_id, make_with_torque_removed, collector, workers, max_timesteps_per_episode, seed)
            iteration = 0
            while True:
                iteration += 1
                # run a bunch of async processes that collect rollouts
                # Eventually this will cause an exception
                rollouts.rollout(timesteps_per_batch, iteration)

    except SuccessfullyCollectedSegments:
        print("Successfully collected %s segments" % len(collector.segments))
        return collector.segments
