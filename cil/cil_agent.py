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
import copy

from agents.dqn_agent import DQNNetworkParameters, DQNAlgorithmParameters
from agents.value_optimization_agent import ValueOptimizationAgent
from architectures.tensorflow_components.heads.categorical_q_head import CategoricalQHeadParameters
from architectures.tensorflow_components.heads.v_head import VHeadParameters
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from base_parameters import AgentParameters, MiddlewareScheme, NetworkParameters, InputEmbedderParameters, \
    EmbedderScheme, AlgorithmParameters
from core_types import StateType, EnvironmentSteps, ActionInfo
from exploration_policies.e_greedy import EGreedyParameters
from memories.experience_replay import ExperienceReplayParameters
from memories.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from schedules import LinearSchedule


class SACVNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True)}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters()]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class SACQNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True),
                                           'action': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters()]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = False
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class SACActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True)}
        self.middleware_parameters = FCMiddlewareParameters(batchnorm=True)
        self.heads_parameters = [DDPGActorHeadParameters()]
        self.loss_weights = [1.0]
        self.rescale_gradient_from_head_by_factor = [1]
        self.optimizer_type = 'Adam'
        self.batch_size = 64
        self.async_training = False
        self.learning_rate = 0.0003
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class CILAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.01


class SACExplorationParameters(EGreedyParameters):
    def __init__(self):
        super().__init__()
        self.epsilon_schedule = LinearSchedule(1, 0.01, 250000//4)  # 250k frames
        self.evaluation_epsilon = 0.001



class CILAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=CILAlgorithmParameters(),
                         exploration=ExplorationParameters(),
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"actor": SACActorNetworkParameters(),
                                   "q": SACQNetworkParameters(),
                                   "v": SACVNetworkParameters()})

    @property
    def path(self):
        return 'sac.sac_agent:SACAgent'


# Soft Actor-Critic Agent: https://arxiv.org/pdf/1801.01290.pdf
class SACAgent(ValueOptimizationAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    # prediction's format is (batch,actions,atoms)
    def get_all_q_values_for_states(self, states: StateType):
        prediction = self.get_prediction(states)
        return self.distribution_prediction_to_q_values(prediction)

    def learn_from_batch(self, batch):
        actor_keys = self.ap.network_wrappers['actor'].input_embedders_parameters.keys()
        q_keys = self.ap.network_wrappers['q'].input_embedders_parameters.keys()
        v_keys = self.ap.network_wrappers['v'].input_embedders_parameters.keys()

        # for the action we actually took, the error is:
        # TD error = r + discount*max(q_st_plus_1) - q_st
        # # for all other actions, the error is 0
        q_inputs = copy.copy(batch.states(q_keys))
        q_inputs['action'] = batch.actions()
        q_st = self.networks['q'].online_network.predict(q_inputs)

        v_st_plus_1, v_TD_targets = self.networks['v'].parallel_prediction([
            (self.networks['v'].target_network, batch.next_states(v_keys)),
            (self.networks['v'].online_network, batch.states(v_keys))
        ])

        # calculate the targets for the V function: V(s_t)=Q(s_t,a_t)-log(pi(a_t,s_t))
        for i in range(self.ap.network_wrappers['q'].batch_size):
            v_TD_targets[i] = q_st[i] - np.log(pi)

        # calculate the targets for the Q function: Q(s_t, a_t)=r+discount*V(s_(t+1))
        TD_errors = []
        for i in range(self.ap.network_wrappers['q'].batch_size):
            new_target = batch.rewards()[i] +\
                         (1.0 - batch.game_overs()[i]) * self.ap.algorithm.discount * v_st_plus_1[i]
            TD_errors.append(np.abs(new_target - q_st[i]))
            q_st[i] = new_target









        result = self.networks['main'].train_and_sync_networks(
            batch.states(network_keys), TD_targets,
            importance_weights=importance_weights,
            additional_fetches=self.networks['main'].online_network.output_heads[0].loss
        )
        total_loss, losses, unclipped_grads = result[:3]


        return total_loss, losses, unclipped_grads


    def choose_action(self, curr_state):
        if not isinstance(self.spaces.action, BoxActionSpace):
            raise ValueError("DDPG works only for continuous control problems")
        # convert to batch so we can run it through the network
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'actor')
        if self.ap.algorithm.use_target_network_for_evaluation:
            actor_network = self.networks['actor'].target_network
        else:
            actor_network = self.networks['actor'].online_network

        action_values = actor_network.predict(tf_input_state).squeeze()

        action = self.exploration_policy.get_action(action_values)

        # bound actions
        # action = self.spaces.action.clip_action_to_space(action)

        # get q value
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'critic')
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        tf_input_state['action'] = action_batch
        q_value = self.networks['critic'].online_network.predict(tf_input_state)[0]
        self.q_values.add_sample(q_value)

        action_info = ActionInfo(action=action,
                                 action_value=q_value,
                                 action_intrinsic_reward=1)  # entropy * alpha

        return action_info
