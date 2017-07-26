from copy import copy

import gym
import numpy as np
from gym.envs import mujoco
from gym.wrappers.time_limit import TimeLimit

class TransparentWrapper(gym.Wrapper):
    """Passes missing attributes through the wrapper stack"""

    def __getattr__(self, attr):
        parent = super()
        if hasattr(parent, attr):
            return getattr(parent, attr)
        if hasattr(self.env, attr):
            return getattr(self.env, attr)
        raise AttributeError(attr)

class MjViewer(TransparentWrapper):
    """Adds a space-efficient human_obs to info that allows rendering videos subsequently"""

    def __init__(self, env, fps=40):
        self.fps = fps
        super().__init__(env)

    def _get_full_obs(self):
        return (copy(self.env.model.data.qpos[:, 0]), copy(self.env.model.data.qvel[:, 0]))

    def _set_full_obs(self, obs):
        qpos, qvel = obs[0], obs[1]
        self.env.set_state(qpos, qvel)

    def render_full_obs(self, full_obs):
        old_obs = self._get_full_obs()
        self._set_full_obs(full_obs)
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        result = ((width, height, 3), data)
        self._set_full_obs(old_obs)
        return result

    def _step(self, a):
        human_obs = self._get_full_obs()
        ob, reward, done, info = self.env._step(a)
        info["human_obs"] = human_obs
        return ob, reward, done, info

class UseReward(TransparentWrapper):
    """Use a reward other than the normal one for an environment.
     We do this because humans cannot see torque penalties
     """

    def __init__(self, env, reward_info_key):
        self.reward_info_key = reward_info_key
        super().__init__(env)

    def _step(self, a):
        ob, reward, done, info = super()._step(a)
        return ob, info[self.reward_info_key], done, info

class NeverDone(TransparentWrapper):
    """Environment that never returns a done signal"""

    def __init__(self, env, bonus=lambda a, data: 0.):
        self.bonus = bonus
        super().__init__(env)

    def _step(self, a):
        ob, reward, done, info = super()._step(a)
        bonus = self.bonus(a, self.env.model.data)
        reward = reward + bonus
        done = False
        return ob, reward, done, info

class TimeLimitTransparent(TimeLimit, TransparentWrapper):
    pass

def limit(env, t):
    return TimeLimitTransparent(env, max_episode_steps=t)

def task_by_name(name, short=False):
    if name == "reacher":
        return reacher(short=short)
    elif name == "humanoid":
        return humanoid()
    elif name == "hopper":
        return hopper(short=short)
    elif name in ["walker"]:
        return walker(short=short)
    elif name == "swimmer":
        return swimmer()
    elif name == "ant":
        return ant()
    elif name in ["cheetah", "halfcheetah"]:
        return cheetah(short=short)
    elif name in ["pendulum"]:
        return pendulum()
    elif name in ["doublependulum"]:
        return double_pendulum()
    else:
        raise ValueError(name)

def make_with_torque_removed(env_id):
    if '-v' in env_id:
        env_id = env_id[:env_id.index('-v')].lower()
    if env_id.startswith('short'):
        env_id = env_id[len('short'):]
        short = True
    else:
        short = False
    return task_by_name(env_id, short)  # Use our task_by_name function to get the env

def get_timesteps_per_episode(env):
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    if hasattr(env, "spec"):
        return env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")
    if hasattr(env, "env"):
        return get_timesteps_per_episode(env.env)
    return None

def simple_reacher():
    return limit(SimpleReacher(), 50)

class SimpleReacher(mujoco.ReacherEnv):
    def _step(self, a):
        ob, _, done, info = super()._step(a)
        return ob, info["reward_dist"], done, info

def reacher(short=False):
    env = mujoco.ReacherEnv()
    env = UseReward(env, reward_info_key="reward_dist")
    env = MjViewer(fps=10, env=env)
    return limit(t=20 if short else 50, env=env)

def hopper(short=False):
    bonus = lambda a, data: (data.qpos[1, 0] - 1) + 1e-3 * np.square(a).sum()
    env = mujoco.HopperEnv()
    env = MjViewer(fps=40, env=env)
    env = NeverDone(bonus=bonus, env=env)
    env = limit(t=300 if short else 1000, env=env)
    return env

def humanoid(standup=True, short=False):
    env = mujoco.HumanoidEnv()
    env = MjViewer(env, fps=40)
    env = UseReward(env, reward_info_key="reward_linvel")
    if standup:
        bonus = lambda a, data: 5 * (data.qpos[2, 0] - 1)
        env = NeverDone(env, bonus=bonus)
    return limit(env, 300 if short else 1000)

def double_pendulum():
    bonus = lambda a, data: 10 * (data.site_xpos[0][2] - 1)
    env = mujoco.InvertedDoublePendulumEnv()
    env = MjViewer(env, fps=10)
    env = NeverDone(env, bonus)
    env = limit(env, 50)
    return env

def pendulum():
    # bonus = lambda a, data: np.concatenate([data.qpos, data.qvel]).ravel()[1] - 1.2
    def bonus(a, data):
        angle = data.qpos[1, 0]
        return -np.square(angle)  # Remove the square of the angle

    env = mujoco.InvertedPendulumEnv()
    env = MjViewer(env, fps=10)
    env = NeverDone(env, bonus)
    env = limit(env, 25)  # Balance for 2.5 seconds
    return env

def cheetah(short=False):
    env = mujoco.HalfCheetahEnv()
    env = UseReward(env, reward_info_key="reward_run")
    env = MjViewer(env, fps=20)
    env = limit(env, 300 if short else 1000)
    return env

def swimmer(short=False):
    env = mujoco.SwimmerEnv()
    env = UseReward(env, reward_info_key="reward_fwd")
    env = MjViewer(env, fps=40)
    env = limit(env, 300 if short else 1000)
    return env

def ant(standup=True, short=False):
    env = mujoco.AntEnv()
    env = UseReward(env, reward_info_key="reward_forward")
    env = MjViewer(env, fps=20)
    if standup:
        bonus = lambda a, data: data.qpos.flat[2] - 1.2
        env = NeverDone(env, bonus)
    env = limit(env, 300 if short else 1000)
    return env

def walker(short=False):
    bonus = lambda a, data: data.qpos[1, 0] - 2.0 + 1e-3 * np.square(a).sum()
    env = mujoco.Walker2dEnv()
    env = MjViewer(env, fps=30)
    env = NeverDone(env, bonus)
    env = limit(env, 300 if short else 1000)
    return env
