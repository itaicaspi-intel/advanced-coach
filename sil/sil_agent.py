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
import scipy.signal
from rl_coach.agents.actor_critic_agent import ActorCriticAgent, ActorCriticAlgorithmParameters, ActorCriticNetworkParameters
from rl_coach.agents.policy_optimization_agent import PolicyOptimizationAgent, PolicyGradientRescaler
from rl_coach.architectures.tensorflow_components.heads.policy_head import PolicyHeadParameters
from rl_coach.architectures.tensorflow_components.heads.v_head import VHeadParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import AlgorithmParameters, NetworkParameters, \
    AgentParameters, InputEmbedderParameters
from rl_coach.core_types import QActionStateValue, Batch, RunPhase
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplay, \
    PrioritizedExperienceReplayParameters
from rl_coach.spaces import DiscreteActionSpace
from rl_coach.utils import last_sample

from rl_coach.logger import screen
from rl_coach.memories.episodic.single_episode_buffer import SingleEpisodeBufferParameters


class SILAlgorithmParameters(ActorCriticAlgorithmParameters):
    def __init__(self):
        super().__init__()
        self.policy_gradient_rescaler = PolicyGradientRescaler.A_VALUE
        self.apply_gradients_every_x_episodes = 5
        self.beta_entropy = 0
        self.num_steps_between_gradient_updates = 5000  # this is called t_max in all the papers
        self.gae_lambda = 0.96
        self.estimate_state_value_using_gae = False
        self.store_transitions_only_when_episodes_are_terminated = True
        self.off_policy_training_steps_per_on_policy_training_steps = 5


class SILNetworkParameters(ActorCriticNetworkParameters):
    def __init__(self):
        super().__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters()}
        self.middleware_parameters = FCMiddlewareParameters()
        self.heads_parameters = [VHeadParameters(), PolicyHeadParameters()]
        self.loss_weights = [0.5, 1.0]
        self.rescale_gradient_from_head_by_factor = [1, 1]
        self.optimizer_type = 'Adam'
        self.clip_gradients = 40.0
        self.batch_size = 32
        self.async_training = False
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

    def learn_from_batch_on_policy(self, batch):
        # batch contains a list of episodes to learn from
        network_keys = self.ap.network_wrappers['main'].input_embedders_parameters.keys()

        # get the values for the current states

        result = self.networks['main'].online_network.predict(batch.states(network_keys))
        current_state_values = result[0]

        self.state_values.add_sample(current_state_values)

        # the targets for the state value estimator
        num_transitions = batch.size
        state_value_head_targets = np.zeros((num_transitions, 1))

        # estimate the advantage function
        action_advantages = np.zeros((num_transitions, 1))

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            if batch.game_overs()[-1]:
                R = 0
            else:
                R = self.networks['main'].online_network.predict(last_sample(batch.next_states(network_keys)))[0]

            for i in reversed(range(num_transitions)):
                R = batch.rewards()[i] + self.ap.algorithm.discount * R
                state_value_head_targets[i] = R
                action_advantages[i] = R - current_state_values[i]

        elif self.policy_gradient_rescaler == PolicyGradientRescaler.GAE:
            # get bootstraps
            bootstrapped_value = self.networks['main'].online_network.predict(last_sample(batch.next_states(network_keys)))[0]
            values = np.append(current_state_values, bootstrapped_value)
            if batch.game_overs()[-1]:
                values[-1] = 0

            # get general discounted returns table
            gae_values, state_value_head_targets = self.get_general_advantage_estimation_values(batch.rewards(), values)
            action_advantages = np.vstack(gae_values)
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        action_advantages = action_advantages.squeeze(axis=-1)
        actions = batch.actions()
        if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) < 2:
            actions = np.expand_dims(actions, -1)

        # train
        result = self.networks['main'].online_network.accumulate_gradients({**batch.states(network_keys),
                                                                            'output_1_0': actions},
                                                                       [state_value_head_targets, action_advantages])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])

        return total_loss, losses, unclipped_grads

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

        # the targets for the state value estimator
        num_transitions = batch.size
        state_value_head_targets = np.maximum(batch.total_returns(expand_dims=True), current_state_values)

        # estimate the advantage function
        action_advantages = np.zeros((num_transitions, 1))

        if self.policy_gradient_rescaler == PolicyGradientRescaler.A_VALUE:
            action_advantages = batch.total_returns() - current_state_values.squeeze()
        else:
            screen.warning("WARNING: The requested policy gradient rescaler is not available")

        # clip negative advantages
        action_advantages = np.clip(action_advantages, 0, np.inf)
        actions = batch.actions()
        if not isinstance(self.spaces.action, DiscreteActionSpace) and len(actions.shape) < 2:
            actions = np.expand_dims(actions, -1)

        # update errors in prioritized replay buffer
        importance_weights = self.update_transition_priorities_and_get_weights(action_advantages, batch)

        # train
        result = self.networks['main'].online_network.accumulate_gradients(
            {**batch.states(network_keys), 'output_1_0': actions}, [state_value_head_targets, action_advantages],
            importance_weights=importance_weights)

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])

        return total_loss, losses, unclipped_grads

    def train(self):
        episode = self.current_episode_buffer  # TODO: in the last transition of the episode, the episode will be taken from the memory which is a bug

        # check if we should calculate gradients or skip
        episode_ended = episode.is_complete
        num_steps_passed_since_last_update = episode.length() - self.last_gradient_update_step_idx
        is_t_max_steps_passed = num_steps_passed_since_last_update >= self.ap.algorithm.num_steps_between_gradient_updates
        if not (is_t_max_steps_passed or episode_ended):
            return 0

        total_loss = 0
        if num_steps_passed_since_last_update > 0:

            # we need to update the returns of the episode until now
            episode.update_returns()

            # get t_max transitions or less if the we got to a terminal state
            # will be used for both actor-critic and vanilla PG.
            # # In order to get full episodes, Vanilla PG will set the end_idx to a very big value.
            transitions = []
            start_idx = self.last_gradient_update_step_idx
            end_idx = episode.length()

            for idx in range(start_idx, end_idx):
                transitions.append(episode.get_transition(idx))
            self.last_gradient_update_step_idx = end_idx

            # update the statistics for the variance reduction techniques
            if self.policy_gradient_rescaler in \
                    [PolicyGradientRescaler.FUTURE_RETURN_NORMALIZED_BY_EPISODE,
                     PolicyGradientRescaler.FUTURE_RETURN_NORMALIZED_BY_TIMESTEP]:
                self.update_episode_statistics(episode)

            # accumulate the gradients and apply them once in every apply_gradients_every_x_episodes episodes
            batch = Batch(transitions)
            total_loss, losses, unclipped_grads = self.learn_from_batch_on_policy(batch)
            if self.current_episode % self.ap.algorithm.apply_gradients_every_x_episodes == 0:
                for network in self.networks.values():
                    network.apply_gradients_and_sync_networks()

            # sil training
            for i in range(self.ap.algorithm.off_policy_training_steps_per_on_policy_training_steps):
                off_policy_loss = self.train_off_policy()

            self.training_iteration += 1

        # move the pointer to the next episode start and discard the episode.
        if episode_ended:
            # we need to remove the episode, because the next training iteration will be called before storing any
            # additional transitions in the memory (we don't store a transition for the first call to observe), so the
            # length of the memory won't be enforced and the old episode won't be removed
            self.last_gradient_update_step_idx = 0

        return total_loss

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
