#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Union

import numpy as np

from agents.dqn_agent import DQNNetworkParameters, DQNAlgorithmParameters
from agents.value_optimization_agent import ValueOptimizationAgent
from architectures.tensorflow_components.heads.categorical_q_head import CategoricalQHeadParameters
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from base_parameters import AgentParameters, MiddlewareScheme
from core_types import StateType, EnvironmentSteps
from exploration_policies.e_greedy import EGreedyParameters
from memories.experience_replay import ExperienceReplayParameters
from memories.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from schedules import LinearSchedule

from rainbow.rainbow_head import RainbowHeadParameters


class RainbowNetworkParameters(DQNNetworkParameters):
    def __init__(self):
        super().__init__()
        self.middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Empty)
        self.heads_parameters = [RainbowHeadParameters()]
        self.optimizer_epsilon = 1.5e-4
        self.learning_rate = 0.00025 / 4


class RainbowAlgorithmParameters(DQNAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.v_min = -10.0
        self.v_max = 10.0
        self.atoms = 51
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(32000 // 4)  # 32k frames
        self.n_steps = 3


class RainbowExplorationParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.epsilon_schedule = LinearSchedule(1, 0.01, 250000//4)  # 250k frames
        self.evaluation_epsilon = 0.001


class RainbowMemoryParameters(PrioritizedExperienceReplayParameters):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.beta = LinearSchedule(0.4, 1, 100000000)


class RainbowAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=RainbowAlgorithmParameters(),
                         exploration=RainbowExplorationParameters(),
                         memory=RainbowMemoryParameters(),
                         networks={"main": RainbowNetworkParameters()})

    @property
    def path(self):
        return 'rainbow.rainbow_agent:RainbowAgent'


# Rainbow Agent - https://arxiv.org/pdf/1710.02298.pdf
# Includes - Double DQN, Categorical DQN, Dueling network, Noisy nets, Prioritized Experience Replay, N-Step Returns
class RainbowAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)
        self.z_values, self.delta_z = np.linspace(self.ap.algorithm.v_min, self.ap.algorithm.v_max,
                                                  self.ap.algorithm.atoms, retstep=True)

    def distribution_prediction_to_q_values(self, prediction):
        return np.dot(prediction, self.z_values)

    # prediction's format is (batch,actions,atoms)
    def get_all_q_values_for_states(self, states: StateType):
        prediction = self.get_prediction(states)
        return self.distribution_prediction_to_q_values(prediction)

    def learn_from_batch(self, batch):
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # TODO: n-step updates
        # TODO: noisy nets

        # for the action we actually took, the error is calculated by the atoms distribution
        # for all other actions, the error is 0
        distributed_q_st_plus_1, TD_targets = self.networks['main'].parallel_prediction([
            (self.networks['main'].target_network, batch.next_states(network_keys)),
            (self.networks['main'].online_network, batch.states(network_keys))
        ])
        online_q_st_plus_1 = self.networks['main'].online_network.predict(batch.next_states(network_keys))

        # DDQN update - select actions for the next state according to the online network,
        # evaluate their value according to the target network
        target_actions = np.argmax(self.distribution_prediction_to_q_values(online_q_st_plus_1), axis=1)

        m = np.zeros((self.ap.network_wrappers['main'].batch_size, self.ap.algorithm.atoms))
        for j in range(self.z_values.size):
            # distributional Bellman operator
            tzj = batch.rewards() + (1.0 - batch.game_overs()) * self.ap.algorithm.discount * self.z_values[j]

            # clip to support
            tzj = np.clip(tzj, self.ap.algorithm.v_min, self.ap.algorithm.v_max)

            # normalize by removing the offset and dividing by the step size
            bj = (tzj - self.ap.algorithm.v_min)/self.delta_z

            # compute upper and lower bounds of bin
            u = (np.ceil(bj)).astype(int)
            l = (np.floor(bj)).astype(int)

            # update bin bound values
            m[:, l] += distributed_q_st_plus_1[:, target_actions, j] * (u - bj)  # dist from upper bound
            m[:, u] += distributed_q_st_plus_1[:, target_actions, j] * (bj - l)  # dist from lower bound

        # total_loss = cross entropy between actual result above and predicted result for the given action
        TD_targets[:, batch.actions()] = m

        # update errors in prioritized replay buffer
        importance_weights = batch.info("weight")

        result = self.networks['main'].train_and_sync_networks(
            batch.states(network_keys), TD_targets,
            importance_weights=importance_weights,
            additional_fetches=self.networks['main'].online_network.output_heads[0].loss
        )
        total_loss, losses, unclipped_grads = result[:3]

        # the errors used for prioritization are the KL losses
        errors = result[-1][0][0, batch.actions()]
        self.memory.update_priorities(batch.info('idx'), errors)

        return total_loss, losses, unclipped_grads

