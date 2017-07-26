import os
import os.path as osp
import random
from collections import deque
from time import time, sleep

import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.segment_sampling import SegmentVideoRecorder
from rl_teacher.segment_sampling import create_segment_q_states
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify

CLIP_LENGTH = 1.5

class TraditionalRLRewardPredictor():
    """Predictor that always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)
        return path["original_rewards"]

    def path_callback(self, path):
        pass

class ComparisonRewardPredictor():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule):
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule

        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = CLIP_LENGTH * env.fps
        self._steps_since_last_training = 0
        self._n_paths_per_predictor_training = 1e2  # How often should we train our predictor?

        # Build and initialize our predictor model
        self.sess = tf.InteractiveSession()
        self.q_state_size = np.product(env.observation_space.shape) + np.product(env.action_space.shape)
        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _predict_rewards(self, segments):
        """
        :param segments: tensor with shape = (batch_size, segment_length, q_state_size)
        :return: tensor with shape = (batch_size, segment_length)
        """
        segment_length = tf.shape(segments)[1]
        batchsize = tf.shape(segments)[0]

        # Temporarily chop up segments into individual q_states
        q_states = tf.reshape(segments, [batchsize * segment_length, self.q_state_size])

        # Run them through our MLP
        rewards = self.mlp.run(q_states)

        # Group the rewards back into their segments
        return tf.reshape(rewards, (batchsize, segment_length))

    def _build_model(self):
        """Our model takes in a vector of q_states from a segment and returns a reward for each one"""
        self.segment_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None, self.q_state_size), name="obs_placeholder")
        self.segment_alt_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None, self.q_state_size), name="obs_placeholder")

        # A vanilla MLP maps a q_state to a reward
        self.mlp = FullyConnectedMLP(self.q_state_size)
        self.q_state_reward_pred = self._predict_rewards(self.segment_placeholder)
        q_state_alt_reward_pred = self._predict_rewards(self.segment_alt_placeholder)

        # We use trajectory segments rather than individual q_states because video clips of segments are easier for
        # humans to evaluate
        segment_reward_pred_left = tf.reduce_sum(self.q_state_reward_pred, axis=1)
        segment_reward_pred_right = tf.reduce_sum(q_state_alt_reward_pred, axis=1)
        reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

        self.loss_op = tf.reduce_mean(data_loss)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=self.global_step)

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        q_state_reward_pred = self.sess.run(self.q_state_reward_pred, feed_dict={
            self.segment_placeholder: np.array([create_segment_q_states(path)]),
            K.learning_phase(): False
        })
        return q_state_reward_pred[0]

    def path_callback(self, path):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        self.agent_logger.log_episode(path)

        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment:
            self.recent_segments.append(segment)

        # If we need more comparisons, then we build them from our recent segments
        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            self.comparison_collector.add_segment_pair(
                random.choice(self.recent_segments),
                random.choice(self.recent_segments))

        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_paths_per_predictor_training):
            self.train_predictor()
            self._steps_since_last_training -= self._steps_since_last_training

    def train_predictor(self):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
        left_q_states = np.asarray([comp['left']['q_states'] for comp in labeled_comparisons])
        right_q_states = np.asarray([comp['right']['q_states'] for comp in labeled_comparisons])

        _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
            self.segment_placeholder: left_q_states,
            self.segment_alt_placeholder: right_q_states,
            self.labels: np.asarray([comp['label'] for comp in labeled_comparisons]),
            K.learning_phase(): True
        })
        self._write_training_summaries(loss)

    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("predictor/loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.last_n_paths
        if recent_paths and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            validation_q_states = np.asarray([create_segment_q_states(path) for path in recent_paths])
            q_state_reward_pred = self.sess.run(self.q_state_reward_pred, feed_dict={
                self.segment_placeholder: validation_q_states,
                K.learning_phase(): False
            })
            ep_reward_pred = np.sum(q_state_reward_pred, axis=1)
            q_state_reward_true = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(q_state_reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", np.corrcoef(ep_reward_true, ep_reward_pred)[0, 1])

        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=2e7, type=int)
    parser.add_argument('-a', '--agent', default="pposgd_mpi", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    args = parser.parse_args()

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.name, int(time()))
    summary_writer = make_summary_writer(run_name)

    env = make_with_torque_removed(env_id)

    num_timesteps = int(args.num_timesteps)
    experiment_name = slugify(args.name)

    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()

        elif args.predictor == "human":
            bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(env_id, experiment_name=experiment_name)
        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(args.seed, env_id, env, n_segments=pretrain_labels * 5)

        # Pull in our pretraining segments
        while len(comparison_collector) < int(pretrain_labels):  # Turn our segments into comparisons
            comparison_collector.add_segment_pair(random.choice(pretrain_segments), random.choice(pretrain_segments))

        # Sleep until the human has labeled most of the pretraining comparisons
        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synth":
                print("%s synthetic labels generated... " % (len(comparison_collector.labeled_comparisons)))
            elif args.predictor == "human":
                print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (
                    len(comparison_collector.labeled_comparisons), pretrain_labels))
                sleep(5)

        # Start the actual training
        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
        )
        for i in range(args.pretrain_iters):
            predictor.train_predictor()  # Train on pretraining labels
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos every so often:
    wrapped_predictor = SegmentVideoRecorder(
        predictor, env, checkpoint_interval=20,
        save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")
    if args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=wrapped_predictor,
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
        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=wrapped_predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
