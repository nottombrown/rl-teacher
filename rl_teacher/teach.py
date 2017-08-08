import os
import argparse
from time import time

import numpy as np
import tensorflow as tf
from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi
from ga3c.Server import Server as Ga3cServer
from ga3c.Config import Config as Ga3cConfig

from rl_teacher.reward_predictors import TraditionalRLRewardPredictor, ComparisonRewardPredictor
from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.video import SegmentVideoRecorder
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef

# TODO: Parameterize this.
CLIP_LENGTH = 1.5

def make_comparison_predictor(env, experiment_name, predictor_type, summary_writer, n_pretrain_labels, n_labels=None):
    agent_logger = AgentLogger(summary_writer)

    if n_labels:
        label_schedule = LabelAnnealer(
            agent_logger,
            final_timesteps=num_timesteps,
            final_labels=n_labels,
            pretrain_labels=n_pretrain_labels)
    else:
        print("No label limit given. We will request one label every few seconds.")
        label_schedule = ConstantLabelSchedule(pretrain_labels=n_pretrain_labels)

    if predictor_type == "synth":
        comparison_collector = SyntheticComparisonCollector()

    elif predictor_type == "human":
        bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        assert bucket, "you must specify a RL_TEACHER_GCS_BUCKET environment variable"
        assert bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
        comparison_collector = HumanComparisonCollector(env, experiment_name=experiment_name)
    else:
        raise ValueError("Bad value for --predictor: %s" % predictor_type)

    return ComparisonRewardPredictor(
        env,
        experiment_name,
        summary_writer,
        comparison_collector=comparison_collector,
        agent_logger=agent_logger,
        label_schedule=label_schedule,
        clip_length=CLIP_LENGTH
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('-r', '--restore', action="store_true")
    args = parser.parse_args()

    print("Setting things up...")
    env_id = args.env_id
    experiment_name = slugify(args.name)
    run_name = "%s/%s-%s" % (env_id, experiment_name, int(time()))
    summary_writer = make_summary_writer(run_name)
    env = make_with_torque_removed(env_id)
    num_timesteps = int(args.num_timesteps)

    os.makedirs('checkpoints/reward_model', exist_ok=True)

    # Make predictor
    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        n_pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4
        predictor = make_comparison_predictor(
            env, experiment_name, args.predictor, summary_writer, n_pretrain_labels, args.n_labels)

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_with_torque_removed, n_desired_segments=n_pretrain_labels * 2,
            clip_length_in_seconds=CLIP_LENGTH, workers=args.workers)

        # Label pretrain segments
        for i in range(n_pretrain_labels):  # Turn our random segments into comparisons
            predictor.comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + n_pretrain_labels])
        predictor.comparison_collector.label_unlabeled_comparisons(goal=n_pretrain_labels, verbose=True)

        if args.restore:
            predictor.load_model_from_checkpoint()
            print("Reward model loaded from checkpoint!")
        else:
            # Pretrain predictor
            for i in range(args.pretrain_iters):
                predictor.train_predictor()  # Train on pretraining labels
                if i % 25 == 0:
                    print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        video_path = os.path.join('/tmp/rl_teacher_vids', run_name)
        predictor = SegmentVideoRecorder(predictor, env, save_dir=video_path, checkpoint_interval=100)

    print("Starting joint training of predictor and agent")
    if args.agent == "ga3c":
        Ga3cConfig.NETWORK_NAME = experiment_name
        Ga3cConfig.SAVE_FREQUENCY = 200
        Ga3cConfig.LOAD_CHECKPOINT = args.restore
        Ga3cConfig.ATARI_GAME = env
        Ga3cConfig.AGENTS = args.workers
        Ga3cServer(predictor).main()
    elif args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=predictor,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=(num_timesteps / 1000),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
        )
    elif args.agent == "pposgd_mpi":
        def make_env():
            return make_with_torque_removed(env_id)

        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
