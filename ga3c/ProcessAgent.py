# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from ga3c.Config import Config
from ga3c.Environment import Environment
from ga3c.Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)

        self._action_onehots = np.eye(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self, iteration):
        self.env.reset()
        done = False
        experiences = []

        path = {
            "obs": [],
            "original_rewards": [],
            "actions": [],
            "human_obs": [],
        }

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            reward, done, human_obs = self.env.step(action)
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, done, human_obs)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                ################################
                #  START REWARD MODIFICATIONS  #
                ################################
                original_rewards = [e.reward for e in experiences]
                if hasattr(Config, "REWARD_MODIFIER"):
                    # Translate the experiences into the "path" that RL-Teacher expects
                    if len(path["obs"]) > 0:
                        # Cut off the first item in the list because it's from an old episode
                        new_experiences = experiences[1:]
                    else:
                        new_experiences = experiences
                    # Ironically we actually want to use the human_obs here
                    # because GA3C uses a 4-frame stack while Teacher learns off only one frame
                    path["obs"] += [e.human_obs for e in new_experiences]
                    path["original_rewards"] += [e.reward for e in new_experiences]
                    path["actions"] += [self._action_onehots[e.action] for e in new_experiences]
                    path["human_obs"] += [e.human_obs for e in new_experiences]

                    # Note: This "prediction" is a different kind than is used in A3C.
                    # This is prediction by RL-Teacher of the "true" reward known by a human.
                    #  TODO SPEED UP!! THIS IS SLOWING THINGS DOWN! BECAUSE IT'S OPERATING ON THE WHOLE PATH!
                    Config.REWARD_MODIFIER.queue.put((self.id, done, path, iteration))
                    path["rewards"] = self.wait_q.get()

                    # Translate new rewards back into the experiences
                    for i in range(len(experiences)):
                        # Work backwards because the path is longer than the experience list, but their ends are synced
                        experiences[-(1 + i)].reward = path["rewards"][-(1 + i)]
                ################################
                #   END REWARD MODIFICATIONS   #
                ################################

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        iteration = 0

        while self.exit_flag.value == 0:
            iteration += 1
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode(iteration):
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
