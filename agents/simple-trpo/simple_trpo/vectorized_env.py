import numpy as np
from multiprocessing import Process, Pipe
import cloudpickle

def env_worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, _ = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VectorizedEnv(object):
    def __init__(self, env_fns):
        """
        :param env_fns: A list of thunks for generating environments
        """
        prototype_env = env_fns[0]() # Construct an env to extract action spaces
        self.action_space = prototype_env.action_space
        self.observation_space = prototype_env.observation_space

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
        self.ps = [Process(target=env_worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def step(self, vectorized_actions):
        """
        :param vectorized_actions: An iterable of actions
        :return: (vectorized_obs, vectorized_rewards, vectorized_dones)
        """
        for remote, action in zip(self.remotes, vectorized_actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)
