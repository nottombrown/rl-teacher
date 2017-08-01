#!/usr/bin/env python
import os.path as osp

import gym
import logging
from pposgd_mpi import bench
from pposgd_mpi.common import logger
from pposgd_mpi.common import set_global_seeds, tf_util as U

def train_pposgd_mpi(make_env, num_timesteps, seed, predictor=None):
    from pposgd_mpi import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = make_env()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=2048*8,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        predictor=predictor
    )
    env.close()

def main():
    def make_env():
        return gym.make('Hopper-v1')

    train_pposgd_mpi(make_env, num_timesteps=1e6, seed=0)

if __name__ == '__main__':
    main()
