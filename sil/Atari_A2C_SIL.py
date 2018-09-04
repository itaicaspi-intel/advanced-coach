from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.environments.environment import SingleLevelSelection, SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.schedules import ConstantSchedule
from sil.sil_agent import SILAgentParameters


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = SILAgentParameters()

agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 5
agent_params.algorithm.beta_entropy = 0.01

agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters()
agent_params.network_wrappers['main'].learning_rate = 0.0007
agent_params.network_wrappers['main'].batch_size = 32
agent_params.network_wrappers['main'].async_training = False
agent_params.network_wrappers['main'].scale_down_gradients_by_number_of_workers_for_sync_training = True

agent_params.memory.shared_memory = True
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)
agent_params.memory.beta = ConstantSchedule(0.1)

agent_params.exploration = CategoricalParameters()

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = False


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
