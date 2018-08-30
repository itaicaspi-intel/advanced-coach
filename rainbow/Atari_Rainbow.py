import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rainbow.rainbow_agent import RainbowAgentParameters
from rl_coach.architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4
from rl_coach.memories.non_episodic.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(1000000)
schedule_params.evaluation_steps = EnvironmentSteps(125000)
schedule_params.heatup_steps = EnvironmentSteps(20000)

################
# Agent Params #
################
agent_params = RainbowAgentParameters()

###############
# Environment #
###############
env_params = Atari()
env_params.level = SingleLevelSelection(atari_deterministic_v4)

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
