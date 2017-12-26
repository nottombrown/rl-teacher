import os

import multiprocess
from time import clock as time

import tensorflow as tf
import numpy as np
import gym

from parallel_trpo.model import TRPO
from parallel_trpo.rollouts import ParallelRollout

from rl_teacher.video import SegmentVideoRecorder

def print_stats(stats):
    for k, v in stats.items():
        if 'time' in k.lower():
            minutes = int(v / 60)
            if minutes:
                v = "{:02d}:{:04.1f}   ".format(minutes, v - minutes * 60)
            else:
                v = "   {:04.2f}  ".format(v)
        elif isinstance(v, int):
            v = str(v) + "     "
        elif isinstance(v, (float, np.floating)):
            v = "{:.4f}".format(v)

        print("{:38} {:>12}".format(k + ":", v))

def train_parallel_trpo(
        env_id,
        predictor,
        make_env=gym.make,
        summary_writer=None,
        workers=1,
        runtime=1800,                           # in seconds, NOT ms
        max_timesteps_per_episode=None,
        timesteps_per_batch=5000,
        max_kl=0.001,
        seed=0,
        discount_factor=0.995,
        cg_damping=0.1,
        load_weights=False,
        save_weights=False,
):
    # Tensorflow is not fork-safe, so we must use spawn instead
    # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405
    # We use multiprocess rather than multiprocessing because Keras sets a multiprocessing context
    if not os.environ.get("SET_PARALLEL_TRPO_START_METHOD"): # Use an env variable to prevent double-setting
        multiprocess.set_start_method('spawn')
        os.environ['SET_PARALLEL_TRPO_START_METHOD'] = "1"

    run_indefinitely = (runtime <= 0)
    print("4 >>> train.py: run_indefinitely=%r, runtime=%s sec" % (run_indefinitely, str(runtime)))

    if max_timesteps_per_episode is None:
        max_timesteps_per_episode = gym.spec(env_id).timestep_limit

    learner = TRPO(
        env_id, make_env,
        max_kl=max_kl,
        discount_factor=discount_factor,
        cg_damping=cg_damping)

    rollouts = ParallelRollout(env_id, make_env, predictor, workers, max_timesteps_per_episode, seed)

    iteration = 0
    start_time = time()     # time since epoch in seconds
    if load_weights:
        learner.load_policy(load_weights)

    while run_indefinitely or time() < start_time + runtime:
        iteration += 1

        # update the weights
        weights = learner.get_policy()
        rollouts.set_policy_weights(weights)

        # run a bunch of async processes that collect rollouts
        paths, rollout_time = rollouts.rollout(timesteps_per_batch)

        # learn from that data
        stats, learn_time = learner.learn(paths)

        # output stats
        print("-------- Iteration %d ----------" % iteration)

        frames_gathered_per_second = stats["Frames gathered"] / rollout_time
        stats["Frames gathered/second"] = int(frames_gathered_per_second)

        stats['Time spent gathering rollouts'] = rollout_time
        stats['Time spent updating weights'] = learn_time

        total_elapsed_seconds = time() - start_time
        stats["Total time"] = total_elapsed_seconds

        print_stats(stats)

        if summary_writer:
            # Log results to tensorboard
            mean_reward = np.mean(np.array([path["original_rewards"].sum() for path in paths]))
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="parallel_trpo/mean_reward", simple_value=mean_reward),
                tf.Summary.Value(tag="parallel_trpo/elapsed_seconds", simple_value=total_elapsed_seconds),
                tf.Summary.Value(tag="parallel_trpo/frames_gathered_per_second", simple_value=frames_gathered_per_second),
            ])
            summary_writer.add_summary(summary, global_step=iteration)

        checkpoint = iteration % 40 == 0
        ### NOTE: checkpoint after 20*200 rollouts
        if checkpoint:
            predictor.save_weights(index=iteration, policy=learner)
                    
            ### NOTE: if we are not learning predictor, then save rollout for review from webapp
            p = predictor.predictor  # className=ComparisonRewardPredictor
            is_learn_predictor = p.comparison_collector.learn_predictor
            if not is_learn_predictor and isinstance(predictor, SegmentVideoRecorder):

                ### ???: why do they use "original_rewards" instead of "rewards"
                sorted_by_rewards = sorted(paths, key=lambda k: np.array(k["original_rewards"]).sum(), reverse=True)
                # save video segment
                best_path = sorted_by_rewards[0]
                worst_path = sorted_by_rewards[-1]
                print("best_path, type=", str(type(best_path)))  # list()
                for key in best_path:
                    print("best_path: " + str(key) + ", type=", str(type(best_path[key])))
                
                p.comparison_collector.add_segment_pair(worst_path, best_path, force_webapp=True)


    rollouts.end()
