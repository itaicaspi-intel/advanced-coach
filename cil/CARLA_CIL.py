import sys
import os

import numpy as np

from cil.cil_agent import CILAgentParameters
from cil.cil_head import RegressionHeadParameters
from rl_coach.architectures.tensorflow_components.architecture import Conv2d, Dense
from rl_coach.architectures.tensorflow_components.middlewares.fc_middleware import FCMiddlewareParameters
from rl_coach.architectures.tensorflow_components.middlewares.middleware import MiddlewareParameters
from rl_coach.environments.carla_environment import CarlaEnvironmentParameters, CameraTypes
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_crop_filter import ObservationCropFilter
from rl_coach.filters.observation.observation_reduction_by_sub_parts_name_filter import ObservationReductionBySubPartsNameFilter
from rl_coach.filters.observation.observation_rescale_to_size_filter import ObservationRescaleToSizeFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import ImageObservationSpace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.architectures.tensorflow_components.embedders.embedder import InputEmbedderParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.environment import MaxDumpMethod, SelectedPhaseOnlyDumpMethod, SingleLevelSelection

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = TrainingSteps(500)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

################
# Agent Params #
################
agent_params = CILAgentParameters()

# forward camera and measurements input
agent_params.network_wrappers['main'].input_embedders_parameters = {
    'forward_camera': InputEmbedderParameters(scheme=[Conv2d([32, 5, 2]),
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
                            batchnorm=True),
     'measurements': InputEmbedderParameters(scheme=[Dense([128]),
                                    Dense([128])])
}

# TODO: batch norm will apply to the fc layers which is not desirable.
# TODO: dropout rate can be configured currently
# TODO: dropout should be configured differenetly per layer [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

# simple fc middleware
agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters(scheme=[Dense([512])])

# output branches
agent_params.network_wrappers['main'].heads_parameters = [
    RegressionHeadParameters(),
    RegressionHeadParameters(),
    RegressionHeadParameters(),
    RegressionHeadParameters()
]
# agent_params.network_wrappers['main'].num_output_head_copies = 4  # follow lane, left, right, straight
agent_params.network_wrappers['main'].rescale_gradient_from_head_by_factor = [1, 1, 1, 1]
agent_params.network_wrappers['main'].loss_weights = [1, 1, 1, 1]
# TODO: there should be another head predicting the speed which is connected directly to the forward camera embedding

agent_params.network_wrappers['main'].batch_size = 120
agent_params.network_wrappers['main'].learning_rate = 0.0002


# crop and rescale the image + use only the forward speed measurement
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

# TODO: if acc > brake => brake = 0. if brake < 0.1 => brake = 0. if speed > 10 and brake = 0 => acc = 0
# TODO: normalize the speed with the maximum speed from the training set speed /= 25 (90 km/h)

agent_params.exploration = AdditiveNoiseParameters()
agent_params.exploration.noise_percentage_schedule = ConstantSchedule(0)
agent_params.exploration.evaluation_noise_percentage = 0

agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(0)

agent_params.memory.load_memory_from_file_path = "/home/cvds_lab/Documents/advanced-coach/carla_train_set_replay_buffer.p"
agent_params.memory.state_key_with_the_class_index = 'high_level_command'
agent_params.memory.num_classes = 4

###############
# Environment #
###############
env_params = CarlaEnvironmentParameters()
env_params.level = 'town1'
env_params.cameras = [CameraTypes.FRONT]
env_params.camera_height = 600
env_params.camera_width = 800
env_params.allow_braking = True
env_params.quality = CarlaEnvironmentParameters.Quality.EPIC

vis_params = VisualizationParameters()
vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]
vis_params.dump_mp4 = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params)
