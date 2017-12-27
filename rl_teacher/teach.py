import os
import os.path as osp
import random
from collections import deque
from time import time, sleep

import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel_trpo
from parallel_trpo.model import TRPO
from pposgd_mpi.run_mujoco import train_pposgd_mpi

from django.conf import settings

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef
from rl_teacher.video import SegmentVideoRecorder
from rl_teacher.utils import tprint

CLIP_LENGTH = 1.5

class TraditionalRLRewardPredictor(object):
    """Predictor that always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)  # <-- This may cause problems in future versions of Teacher.
        return path["original_rewards"]

    def path_callback(self, path):
        pass

class ComparisonRewardPredictor():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule, experiment_name=None):
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule
        self.experiment_name = experiment_name

        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = CLIP_LENGTH * env.fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = 1e2  # How often should we train our predictor?
        self._elapsed_predictor_training_iters = 0

        # Build and initialize our predictor model
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.InteractiveSession(config=config)
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.keras_NN = None
        # self.predictor_weights: file_name for last save of Predictor weights
        self.predictor_weights = None
        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        batchsize = tf.shape(obs_segments)[0]
        segment_length = tf.shape(obs_segments)[1]

        # Temporarily chop up segments into individual observations and actions
        obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
        acts = tf.reshape(act_segments, (-1,) + self.act_shape)

        # Run them through our neural network
        rewards = network.run(obs, acts)

        # Group the rewards back into their segments
        return tf.reshape(rewards, (batchsize, segment_length))

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up observation placeholders
        self.segment_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")
        self.segment_alt_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="alt_obs_placeholder")

        self.segment_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
        self.segment_alt_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="alt_act_placeholder")


        # A vanilla multi-layer perceptron maps a (state, action) pair to a reward (Q-value)
        self.keras_NN = mlp = FullyConnectedMLP(self.obs_shape, self.act_shape)

        self.q_value = self._predict_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, mlp)
        alt_q_value = self._predict_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, mlp)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = tf.reduce_sum(self.q_value, axis=1)
        segment_reward_pred_right = tf.reduce_sum(alt_q_value, axis=1)
        reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

        self.loss_op = tf.reduce_mean(data_loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=global_step)
        #
        # TODO: using Q-value function approx, qhat(S,A,w), w=weights for Q-value function
        # ???: where is this?
        # 
        return tf.get_default_graph()

    def save_weights(self, index=None, policy=None):
        if not index:
            index = int(time())
        file_name = self.experiment_name

        if self.comparison_collector.learn_predictor and self.keras_NN:
            predictor_name = "%s.predictor_weights.%d.h5" % (self.experiment_name, index)
            # save weights for Predictor
            self.keras_NN.model.save_weights(predictor_name)
            self.predictor_weights = predictor_name

        if policy and isinstance(policy, TRPO):
            policy_name = "%s.policy_weights.%d.ckpt" % (self.experiment_name, index)
            # save TRPO weights for reloading
            policy_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +"/"+ policy_name
            save_path = policy.save_policy(file_name=policy_name)
        
        return predictor_name


    def load_weights(self, file_name):
        if self.keras_NN:
            assert file_name and file_name.endswith(".h5"), "ERROR: weight file must end with `.h5`"
            self.keras_NN.model.load_weights(file_name)
            return file_name

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: np.asarray([path["obs"]]),
                self.segment_act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return q_value[0]

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
        # -- unless we are using a pre-trained reward predictor
        is_learn_predictor = self.comparison_collector.learn_predictor
        if (
            is_learn_predictor 
            and self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training)
        ):
            #
            # BUG: during learning phase
            # ValueError: Cannot feed value of shape (0,) for Tensor 'alt_act_placeholder:0', which has shape '(?, ?, 6)'
            # FIX: add some pretraining data. caused by [] dataset
            # 
            self.train_predictor()
            self._steps_since_last_training -= self._steps_since_last_training

    def train_predictor(self, save_weights=False):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
        if len(labeled_comparisons)==0:
            # BUG ValueError: Cannot feed value of shape (0,) for Tensor 'alt_act_placeholder:0', which has shape '(?, ?, 6)'
            return

        # tprint("5 >>> labeled_comparisons, len=", len(labeled_comparisons))
        left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
        left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
        right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
        right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
        # tprint("5 >>> observations, actions, len=", len(left_obs), len(left_acts), len(right_obs), len(right_acts))
        labels = np.asarray([comp['label'] for comp in labeled_comparisons])
        # tprint("5 >>> labels, len=", len(labels))

        with self.graph.as_default():
            _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
                self.segment_obs_placeholder: left_obs,
                self.segment_act_placeholder: left_acts,
                self.segment_alt_obs_placeholder: right_obs,
                self.segment_alt_act_placeholder: right_acts,
                self.labels: labels,
                K.learning_phase(): True
            })
            self._elapsed_predictor_training_iters += 1
            self._write_training_summaries(loss)

        if save_weights:
            self.save_weights()

    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("predictor/loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            validation_obs = np.asarray([path["obs"] for path in recent_paths])
            validation_acts = np.asarray([path["actions"] for path in recent_paths])
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: validation_obs,
                self.segment_act_placeholder: validation_acts,
                K.learning_phase(): False
            })
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
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
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="parallel_trpo", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('--load_predictor_weights', default=None, type=str)
    parser.add_argument('--train_predictor', default=None, type=int)
    parser.add_argument('--load_policy_weights', default=None, type=str)
    parser.add_argument('--content_server', default=None, type=str)


    args = parser.parse_args()

    print("Setting things up...")

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

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels or 1

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()

        elif args.predictor == "human":
            bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            if args.content_server != "local":
                assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"

            comparison_collector = HumanComparisonCollector(
                env_id
                , experiment_name=experiment_name
                , force=True
                , reset=False
                )

        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
            experiment_name=experiment_name,
        )




        if args.load_predictor_weights:
            ### load pre-trained weights for reward predictor 
            loadFile = predictor.load_weights(args.load_predictor_weights)
            tprint("0 >>> Loading pre-trained weights for reward predictor, file=", loadFile)

            ### NOTE: continue to learn the reward predictor when using pre-trained weights???
            comparison_collector.learn_predictor = not args.load_predictor_weights if args.train_predictor is None else args.train_predictor
            print("0 >>> train reward predictor=", comparison_collector.learn_predictor)

        else:
            ### NOTE: randomly initialize reward predictor and learn from pretrain_labels
            pretrain_labels = max(20, pretrain_labels)
            tprint("0 >>> Starting random rollouts to generate pretraining segments. No learning will take place...")
            pretrain_segments = segments_from_rand_rollout(
                env_id, make_with_torque_removed, n_desired_segments=pretrain_labels * 2,
                clip_length_in_seconds=CLIP_LENGTH, workers=args.workers)
            for i in range(pretrain_labels):  # Turn our random segments into comparisons
                comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])

            # Sleep until the human has labeled most of the pretraining comparisons
            tprint("1 >>> wait for pretraining labels")
            while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):         
                comparison_collector.label_unlabeled_comparisons()
                if args.predictor == "synth":
                    print("%s synthetic labels generated... " % (len(comparison_collector.labeled_comparisons)))
                elif args.predictor == "human":
                    print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (len(comparison_collector.labeled_comparisons), pretrain_labels))
                    sleep(5)

            # Start the actual learning from pretraining
            tprint("2 >>> predictor pretraining, iters=%s " % (args.pretrain_iters))
            for i in range(args.pretrain_iters):
                name = experiment_name + "-pretrain"
                predictor.train_predictor(save_weights=False)  # Train on pretraining labels
                if i % 100 == 0:
                    tprint("2 >>> %s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

            # save predictor state to enable rerun from here
            predictor.save_weights()  # Train on pretraining labels
            tprint("2 >>> Saving pretraining weights to file=", predictor.predictor_weights)


        ### NOTE: load pre-trained weights for agent, TRPO policy predictor
        policyFile = args.load_policy_weights or None


        if args.content_server == "local":
            settings.MEDIA_URL = '/media/'
            settings.DEBUG = True
        else: 
            # default
            settings.MEDIA_URL = 'https://storage.googleapis.com/'


    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        predictor = SegmentVideoRecorder(predictor, env, save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    tprint("3 >>> Starting joint training of predictor and agent, runtime=", num_timesteps)
    if args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=predictor,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=(num_timesteps),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
            load_weights=policyFile,
            save_weights=True,
        )
    elif args.agent == "pposgd_mpi":
        def make_env():
            return make_with_torque_removed(env_id)

        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
