import numpy as np
import tensorflow as tf
from parallel_trpo.train import train_parallel

from rl_teacher.envs import get_timesteps_per_episode, make_with_torque_removed
from rl_teacher.video import write_segment_to_video

def create_segment_q_states(segment):
    obs_Ds = segment["obs"]
    act_Ds = segment["actions"]
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
        if k in ['obs', 'actions', 'original_rewards', 'human_obs']}

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
        self.predictor.path_callback(path, iteration)

    def predict_reward(self, path):
        return self.predictor.predict_reward(path)

class RandomRolloutSegmentCollector(object):
    def __init__(self, n_desired_segments, fps):
        self.n_desired_segments = n_desired_segments
        self.fps = fps
        self.segments = []

    def predict_reward(self, path):
        epsilon = 1e-9  # Reward is unused during random rollout so we return a tiny value for each q_state
        return np.ones(len(path["obs"])) * epsilon

    def path_callback(self, path, iteration):
        clip_length_in_seconds = 1.5
        segment = sample_segment_from_path(path, int(clip_length_in_seconds * self.fps))
        if segment:
            self.segments.append(segment)

        if len(self.segments) % 10 == 0 and len(self.segments) > 0:
            print("Collected %s/%s segments" % (len(self.segments), self.n_desired_segments))

        if len(self.segments) >= self.n_desired_segments:
            raise SuccessfullyCollectedSegments()

def random_action(env, ob):
    """ Pick an action by uniformly sampling the environment's action space. """
    return env.action_space.sample()

def do_rollout(env, action_function):
    """ Builds a path by running through an environment using a provided function to select actions. """
    obs, rewards, actions, human_obs = [], [], [], []
    max_timesteps_per_episode = get_timesteps_per_episode(env)
    ob = env.reset()
    # Primary environment loop
    for i in range(max_timesteps_per_episode):
        action = action_function(env, ob)
        obs.append(ob)
        actions.append(action)
        ob, rew, done, info = env.step(action)
        rewards.append(rew)
        human_obs.append(info.get("human_obs"))
        if done:
            break
    # Build path dictionary
    path = {
        "obs": np.array(obs),
        "rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs)}
    return path

def segments_from_rand_rollout(env, n_desired_segments, clip_length_in_seconds):
    segments = []
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action)

        segment = sample_segment_from_path(path, int(clip_length_in_seconds * env.fps))
        if segment:
            segments.append(segment)

        if len(segments) % 10 == 0 and len(segments) > 0:
            print("Collected %s/%s segments" % (len(segments), n_desired_segments))

    print("Successfully collected %s segments" % len(segments))
    return segments
