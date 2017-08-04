import math
from multiprocessing import Pool
import numpy as np
import gym.spaces.prng as space_prng

from rl_teacher.envs import get_timesteps_per_episode

def _slice_path(path, segment_length, start_pos=0):
    return {
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', "actions", 'original_rewards', 'human_obs']}

def create_segment_q_states(segment):
    obs_Ds = segment["obs"]
    act_Ds = segment["actions"]
    return np.concatenate([obs_Ds, act_Ds], axis=1)

def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    # Add q_states
    segment["q_states"] = create_segment_q_states(segment)
    return segment

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
        "original_rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs)}
    return path

def basic_segments_from_rand_rollout(
    env_id, make_env, n_desired_segments, clip_length_in_seconds,
    # These are only for use with multiprocessing
    seed=0, _verbose=True, _multiplier=1
):
    """ Generate a list of path segments by doing random rollouts. No multiprocessing. """
    segments = []
    env = make_env(env_id)
    env.seed(seed)
    space_prng.seed(seed)
    segment_length = int(clip_length_in_seconds * env.fps)
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action)
        # Calculate the number of segments to sample from the path
        # Such that the probability of sampling the same part twice is fairly low.
        segments_for_this_path = max(1, int(0.25 * len(path["obs"]) / segment_length))
        for _ in range(segments_for_this_path):
            segment = sample_segment_from_path(path, segment_length)
            if segment:
                segments.append(segment)

            if _verbose and len(segments) % 10 == 0 and len(segments) > 0:
                print("Collected %s/%s segments" % (len(segments) * _multiplier, n_desired_segments * _multiplier))

    if _verbose:
        print("Successfully collected %s segments" % (len(segments) * _multiplier))
    return segments

def segments_from_rand_rollout(env_id, make_env, n_desired_segments, clip_length_in_seconds, workers):
    """ Generate a list of path segments by doing random rollouts. Can use multiple processes. """
    if workers < 2:  # Default to basic segment collection
        return basic_segments_from_rand_rollout(env_id, make_env, n_desired_segments, clip_length_in_seconds)

    pool = Pool(processes=workers)
    segments_per_worker = int(math.ceil(n_desired_segments / workers))
    # One job per worker. Only the first worker is verbose.
    jobs = [
        (env_id, make_env, segments_per_worker, clip_length_in_seconds, i, i == 0, workers)
        for i in range(workers)]
    results = pool.starmap(basic_segments_from_rand_rollout, jobs)
    pool.close()
    return [segment for sublist in results for segment in sublist]
