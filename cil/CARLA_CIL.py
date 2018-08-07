import sys
import os

import numpy as np

from architectures.tensorflow_components.architecture import Conv2d, Dense
from architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from architectures.tensorflow_components.middlewares.middleware import MiddlewareParameters
from environments.carla_environment import CarlaEnvironmentParameters, CameraTypes
from filters.filter import InputFilter
from filters.observation.observation_crop_filter import ObservationCropFilter
from filters.observation.observation_reduction_by_sub_parts_name_filter import ObservationReductionBySubPartsNameFilter
from filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from spaces import ImageObservationSpace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rainbow.rainbow_agent import RainbowAgentParameters
from architectures.tensorflow_components.heads.dueling_q_head import DuelingQHeadParameters
from graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from graph_managers.graph_manager import ScheduleParameters
from base_parameters import VisualizationParameters, InputEmbedderParameters
from core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection
from environments.gym_environment import Atari, atari_deterministic_v4
from memories.prioritized_experience_replay import PrioritizedExperienceReplayParameters
from schedules import LinearSchedule

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

agent_params.network_wrappers['main'].input_embedders_parameters['forward_camera'] = \
    InputEmbedderParameters(scheme=[Conv2d([32, 5, 2]),
                                    Conv2d([32, 3, 1]),
                                    Conv2d([64, 3, 2]),
                                    Conv2d([64, 3, 1]),
                                    Conv2d([128, 3, 2]),
                                    Conv2d([128, 3, 1]),
                                    Conv2d([256, 3, 1]),
                                    Conv2d([256, 3, 1]),
                                    Dense([512]),
                                    Dense([512])],
                            dropout=True,
                            batchnorm=True)
# TODO: batch norm will apply to the fc layers which is not desirable.
# TODO: what about flattening?
# TODO: dropout rate can be configured currently
# TODO: dropout should be configured differenetly per layer [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

agent_params.network_wrappers['main'].input_embedders_parameters['measurements'] = \
    InputEmbedderParameters(scheme=[Dense([128]),
                                    Dense([128])])

agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters(scheme=[Dense([512])])

# TODO: head structure and loss

agent_params.network_wrappers['main'].batch_size = 120
agent_params.network_wrappers['main'].learning_rate = 0.0002

agent_params.input_filter = InputFilter()
agent_params.input_filter.add_observation_filter('forward_camera', 'cropping',
                                                 ObservationCropFilter(crop_low=np.array([115, 0, 0]),
                                                                       crop_high=np.array([510, -1, -1])))
agent_params.input_filter.add_observation_filter('forward_camera', 'rescale',
                                                 ObservationRescaleToSizeFilter(
                                                     ImageObservationSpace(np.array([88, 200, 3]), high=255)))
agent_params.input_filter.add_observation_filter('forward_camera', 'to_uint8', ObservationToUInt8Filter(0, 255))
agent_params.input_filter.add_observation_filter(
    'measurements', 'select_speed',
    ObservationReductionBySubPartsNameFilter(
        ["forward_speed"], reduction_method=ObservationReductionBySubPartsNameFilter.ReductionMethod.Keep))

# TODO: batches should contain an equal number of samples from each command
# TODO: filter only the forward speed  V
# TODO: slice image between 115-510    V
# TODO: rescale image to 200x88        V
# TODO: image rescaled to 0-1          V

# TODO: if acc > brake => brake = 0. if brake < 0.1 => brake = 0. if speed > 10 and brake = 0 => acc = 0

# TODO: normalize the speed with the maximum speed from the training set speed /= 25


###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.level = 'town1'
env_params.cameras = [CameraTypes.FRONT]

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
