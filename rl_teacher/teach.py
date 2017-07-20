import os
import os.path as osp
import random
import uuid
from collections import deque
from time import time, sleep

import multiprocessing
import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel
from ga3c.Server import Server as Ga3cServer
from ga3c.Config import Config as Ga3cConfig

from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.nn import two_layer_fc_net
from rl_teacher.segment_sampling import SegmentVideoRecorder
from rl_teacher.segment_sampling import create_segment_q_states
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify
from rl_teacher.video import write_segment_to_video, upload_to_gcs

CLIP_LENGTH = 1.5

class TraditionalRLRewardPredictor():
    """Always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        # self.agent_logger.log_episode(path)  <-- This causes problems for GA3C
        return path["original_rewards"]

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
        self._started_at = None # Don't initialize until we call n_desired_labels
        self._seconds_between_labels = seconds_between_labels
        self._pretrain_labels = pretrain_labels

    @property
    def n_desired_labels(self):
        if self._started_at is None:
            self._started_at = time()
        return self._pretrain_labels + (time() - self._started_at) / self._seconds_between_labels

class SyntheticComparisonCollector(object):
    def __init__(self):
        self._comparisons = []

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1


def _write_and_upload_video(env_id, gcs_path, local_path, segment):
    env = make_with_torque_removed(env_id)
    write_segment_to_video(segment, fname=local_path, env=env)
    upload_to_gcs(local_path, gcs_path)

class HumanComparisonCollector():
    def __init__(self, env_id, experiment_name):
        from human_feedback_api import Comparison

        self._comparisons = []
        self.env_id = env_id
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(4)

        if Comparison.objects.filter(experiment_name=experiment_name).count() > 0:
            raise EnvironmentError("Existing experiment named %s! Pick a new experiment name." % experiment_name)

    def convert_segment_to_media_url(self, comparison_uuid, side, segment):
        tmp_media_dir = '/tmp/rl_teacher_media'
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = osp.join(tmp_media_dir, media_id)
        gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        gcs_path = osp.join(gcs_bucket, media_id)
        self._upload_workers.apply_async(_write_and_upload_video, (self.env_id, gcs_path, local_path, segment))

        media_url = "https://storage.googleapis.com/%s/%s" % (gcs_bucket.lstrip("gs://"), media_id)
        return media_url

    def _create_comparison_in_webapp(self, left_seg, right_seg):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(comparison_uuid, 'left', left_seg),
            media_url_2=self.convert_segment_to_media_url(comparison_uuid, 'right', right_seg),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""

        comparison_id = self._create_comparison_in_webapp(left_seg, right_seg)
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "id": comparison_id,
            "label": None
        }

        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        from human_feedback_api import Comparison

        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is no response yet, so we just wait

class ComparisonRewardPredictor():
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
        q_state_size = np.product(env.observation_space.shape) + np.product(env.action_space.shape)
        self.build_model(q_state_size)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, q_state_size):
        """Our model takes in a vector of q_states from a segment and returns a reward for each one"""
        self.segment_placeholder_Ds = tf.placeholder(
            dtype=tf.float32, shape=(None, None, q_state_size), name="obs_placeholder")
        self.segment_alt_placeholder_Ds = tf.placeholder(
            dtype=tf.float32, shape=(None, None, q_state_size), name="obs_placeholder")

        batchsize = tf.shape(self.segment_placeholder_Ds)[0]
        joint_batchsize = batchsize * 2
        segment_length = tf.shape(self.segment_placeholder_Ds)[1]

        # Join both segment placeholder inputs (left and right)
        segments_Ds = tf.concat([self.segment_placeholder_Ds, self.segment_alt_placeholder_Ds], axis=0)

        # Temporarily chop up segments into individual q_states
        q_state_Dbs = tf.reshape(segments_Ds, [joint_batchsize * segment_length, q_state_size])

        # A vanilla MLP maps a q_state to a reward
        rew_Dbs = two_layer_fc_net(q_state_Dbs)

        # Group the q_states back into their segments
        self.q_state_reward_pred_Ds = tf.reshape(rew_Dbs, (joint_batchsize, segment_length))

        # We use trajectory segments rather than individual q_states because video clips of segments are easier for
        # humans to evaluate
        segment_reward_pred = tf.reduce_sum(self.q_state_reward_pred_Ds, axis=1)

        # TODO; refactor this so we just pass our values through the graph twice, pull the nn weights into an
        # Separate left and right back apart
        segment_reward_pred_left = tf.slice(segment_reward_pred, [0], [batchsize])
        segment_reward_pred_right = tf.slice(segment_reward_pred, [batchsize], [batchsize])

        # Compute our reward
        reward_logits_D2 = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits_D2, labels=self.labels)

        self.loss_op = tf.reduce_mean(data_loss)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=self.global_step)

        with tf.name_scope('predictor'):
            tf.summary.scalar("reward_pred_loss", self.loss_op)
            tf.summary.scalar("predicted_reward_per_step", tf.reduce_mean(self.q_state_reward_pred_Ds))
            # true_reward_per_step = tf.reduce_mean(self.segment_reward_placeholder) / tf.cast(segment_length, tf.float32)
            # tf.summary.scalar("true_reward_per_step", true_reward_per_step)

        self.summary_op = tf.summary.merge_all()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
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

        # Run inference through our predictor
        segments = np.array([create_segment_q_states(path)])
        reward_pred_Ds = self.sess.run(self.q_state_reward_pred_Ds, feed_dict={
            self.segment_placeholder_Ds: segments,
            self.segment_alt_placeholder_Ds: np.zeros(segments.shape),  # We don't use the alt placeholder for inference
            K.learning_phase(): False
        })
        return reward_pred_Ds[0]

    def train_predictor(self):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
        left_q_states = np.asarray([comp['left']['q_states'] for comp in labeled_comparisons])
        right_q_states = np.asarray([comp['right']['q_states'] for comp in labeled_comparisons])

        _, summary, loss = self.sess.run([self.train_op, self.summary_op, self.loss_op], feed_dict={
            self.segment_placeholder_Ds: left_q_states,
            self.segment_alt_placeholder_Ds: right_q_states,
            self.labels: np.asarray([comp['label'] for comp in labeled_comparisons]),
            K.learning_phase(): True
        })

        # Write summaries
        self.summary_writer.add_summary(summary, self.agent_logger.summary_step)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple("labels/labeled_comparisons", len(labeled_comparisons))

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.last_n_paths
        if recent_paths:
            validation_q_states = np.asarray([create_segment_q_states(path) for path in recent_paths])
            reward_pred_Ds = self.sess.run(self.q_state_reward_pred_Ds, feed_dict={
                self.segment_placeholder_Ds: validation_q_states,
                self.segment_alt_placeholder_Ds: np.zeros(validation_q_states.shape),
                # We don't use the alt placeholder
                K.learning_phase(): False
            })
            reward_pred_Ds = reward_pred_Ds[:len(validation_q_states)]

            ep_reward_pred = np.sum(reward_pred_Ds, axis=1)
            self.agent_logger.log_simple("validation/pred_per_episode", np.mean(ep_reward_pred))
            reward_true_Ds = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(reward_true_Ds, axis=1)

            self.agent_logger.log_simple("validation/true_per_episode", np.mean(ep_reward_true))
            self.agent_logger.log_simple("validation/correlations", np.corrcoef(ep_reward_true, ep_reward_pred)[0, 1])

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
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
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
            label_schedule = LabelAnnealer(agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        print("Starting pretraining of predictor")
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
    wrapped_predictor = SegmentVideoRecorder(predictor, env, checkpoint_interval=2,
        save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")
    if args.agent == "ga3c":
        Ga3cConfig.ATARI_GAME = env
        Ga3cConfig.REWARD_MODIFIER = wrapped_predictor
        Ga3cServer().main()
    elif args.agent == "parallel_trpo":
        train_parallel(
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
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
