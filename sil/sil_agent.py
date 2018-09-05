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
from rl_coach.utils import force_list

from rl_coach.agents.actor_critic_agent import ActorCriticAgent, ActorCriticAlgorithmParameters, \
    ActorCriticNetworkParameters
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from rl_coach.architectures.tensorflow_components.heads.v_head import VHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AgentParameters
from rl_coach.core_types import Batch
from rl_coach.logger import screen
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplay, \
    PrioritizedExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace


class SILAlgorithmParameters(ActorCriticAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.policy_gradient_rescaler = PolicyGradientRescaler.A_VALUE
        self.apply_gradients_every_x_episodes = 1
        self.beta_entropy = 0.01  # only used for A2C training, when training with SIL it is overwritten by 0
        self.num_steps_between_gradient_updates = 5  # this is called t_max in all the papers
        self.gae_lambda = 0.96
        self.estimate_state_value_using_gae = False
        self.store_transitions_only_when_episodes_are_terminated = True  # since we want to calculate the returns before
                                                                         # adding it to the replay buffer
        self.off_policy_training_steps_per_on_policy_training_steps = 4


class SILNetworkParameters(ActorCriticNetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters(), PolicyHeadParameters()]
        self.loss_weights = [0.5, 1.0]
        self.sil_loss_weights = [0.5*0.01, 1.0]  # called beta^SIL in the paper
        self.rescale_gradient_from_head_by_factor = [1, 1]
        self.optimizer_type = 'Adam'
        self.clip_gradients = 40.0
        self.batch_size = 32  # = 512 / 16 workers (since training is synchronous)
        self.async_training = False  # A2C
        self.shared_optimizer = True


class SILAgentParameters(AgentParameters):
    def __init__(self):
        super().__init__(algorithm=SILAlgorithmParameters(),
                         exploration=None, #TODO this should be different for continuous (ContinuousEntropyExploration)
                                           #  and discrete (CategoricalExploration) action spaces. how to deal with that?
                         memory=PrioritizedExperienceReplayParameters(),
                         networks={"main": SILNetworkParameters()})

    @property
    def path(self):
        return 'sil.sil_agent:SILAgent'


# Self Imitation Learning
class SILAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    # needed here since it is only implemented for ValueOptimizationAgent
    def update_transition_priorities_and_get_weights(self, TD_errors, batch):
        # update errors in prioritized replay buffer
        importance_weights = None
        if isinstance(self.memory, PrioritizedExperienceReplay):
            self.call_memory('update_priorities', (batch.info('idx'), TD_errors))
            importance_weights = batch.info('weight')
        return importance_weights

    def learn_from_batch_off_policy(self, batch):
        # batch contains a list of episodes to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # get the values for the current states
        result = self.networks['main'].online_network.predict(batch.states(network_keys))
        current_state_values = result[0]
        self.state_values.add_sample(current_state_values)

        # the targets for the state value estimator are max(R, V) which is the same as clipping the error to > 0
        num_transitions = batch.size
        state_value_head_targets = np.maximum(batch.total_returns(expand_dims=True), current_state_values)

        # estimate the advantage function
        action_advantages = np.zeros((num_transitions, 1))

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            action_advantages = batch.total_returns() - current_state_values.squeeze()
            # clip negative advantages to get the SIL rescaler (R - V)+
            action_advantages = np.clip(action_advantages, 0, np.inf)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # extract action indices
        actions = batch.actions()
        if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) < 2:
            actions = np.expand_dims(actions, -1)

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(action_advantages, batch)

        # train
        result = self.networks['main'].train_and_sync_networks(
            {**batch.states(network_keys), 'output_1_0': actions}, [state_value_head_targets, action_advantages],
            importance_weights=importance_weights)

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])

        return total_loss, losses, unclipped_grads

    def post_training_commands(self):
        # remove entropy regularization
        self.networks['main'].online_network.set_variable_value(
            self.networks['main'].online_network.output_heads[1].set_beta, 0,
            self.networks['main'].online_network.output_heads[1].beta_placeholder
        )

        # set the loss weights to the SIL loss weights
        for output_head_idx, output_head in enumerate(self.networks['main'].online_network.output_heads):
            self.networks['main'].online_network.set_variable_value(
                output_head.set_loss_weight,
                force_list(self.ap.network_wrappers['main'].sil_loss_weights[output_head_idx]),
                output_head.loss_weight_placeholder
            )

        # sil training
        for i in range(self.ap.algorithm.off_policy_training_steps_per_on_policy_training_steps):
            off_policy_loss = self.train_off_policy()

        # add back entropy regularization
        self.networks['main'].online_network.set_variable_value(
            self.networks['main'].online_network.output_heads[1].set_beta,
            self.ap.algorithm.beta_entropy,
            self.networks['main'].online_network.output_heads[1].beta_placeholder
        )

        # recover the regular loss weights
        for output_head_idx, output_head in enumerate(self.networks['main'].online_network.output_heads):
            self.networks['main'].online_network.set_variable_value(
                output_head.set_loss_weight,
                force_list(self.ap.network_wrappers['main'].loss_weights[output_head_idx]),
                output_head.loss_weight_placeholder
            )

    def train_off_policy(self):
        loss = 0

        # TODO: this should be network dependent!
        network_parameters = list(self.ap.network_wrappers.values())[0]

        # update counters
        self.training_iteration += 1

        # sample a batch and train on it
        batch = self.call_memory('sample', network_parameters.batch_size)
        if self.pre_network_filter is not None:
            batch = self.pre_network_filter.filter(batch, update_internal_state=False, deep_copy=False)

        # if the batch returned empty then there are not enough samples in the replay buffer -> skip
        # training step
        if len(batch) > 0:
            # train
            batch = Batch(batch)
            total_loss, losses, unclipped_grads = self.learn_from_batch_off_policy(batch)
            loss += total_loss
            self.unclipped_grads.add_sample(unclipped_grads)
            self.loss.add_sample(loss)

        return loss
