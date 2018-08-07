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
from rl_coach.core_types import QActionStateValue, Batch
from rl_coach.memories.episodic.episodic_experience_replay import EpisodicExperienceReplayParameters
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
                         memory=EpisodicExperienceReplayParameters(),
                         networks={"main": SILNetworkParameters()})

    @property
    def path(self):
        return 'sil.sil_agent:SILAgent'


# Self Imitation Learning
class SILAgent(ActorCriticAgent):
    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent']=None):
        super().__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
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

        # train
        result = self.networks['main'].online_network.accumulate_gradients(
            {**batch.states(network_keys), 'output_1_0': actions}, [state_value_head_targets, action_advantages])

        # logging
        total_loss, losses, unclipped_grads = result[:3]
        self.action_advantages.add_sample(action_advantages)
        self.unclipped_grads.add_sample(unclipped_grads)
        self.value_loss.add_sample(losses[0])
        self.policy_loss.add_sample(losses[1])

        return total_loss, losses, unclipped_grads

    def train(self):
        loss = 0
        if self._should_train():
            for training_step in range(self.ap.algorithm.num_consecutive_training_steps):
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
                    total_loss, losses, unclipped_grads = self.learn_from_batch(batch)
                    loss += total_loss
                    self.unclipped_grads.add_sample(unclipped_grads)

                    # TODO: why is this done here? it also uses tensorflow directly which is problematic
                    # decay learning rate
                    if network_parameters.learning_rate_decay_rate != 0:
                        self.curr_learning_rate.add_sample(self.networks['main'].sess.run(
                            self.networks['main'].online_network.current_learning_rate))
                    else:
                        self.curr_learning_rate.add_sample(network_parameters.learning_rate)

                    if any([network.has_target for network in self.networks.values()]) \
                            and self._should_update_online_weights_to_target():
                        for network in self.networks.values():
                            network.update_target_network(self.ap.algorithm.rate_for_copying_weights_to_target)

                        self.agent_logger.create_signal_value('Update Target Network', 1)
                    else:
                        self.agent_logger.create_signal_value('Update Target Network', 0, overwrite=False)

                    self.loss.add_sample(loss)

                    if self.imitation:
                        self.log_to_screen()

            # run additional commands after the training is done
            self.post_training_commands()

        return loss
