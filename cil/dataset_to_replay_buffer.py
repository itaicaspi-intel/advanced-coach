import h5py
import os
import sys
import numpy as np
from rl_coach.core_types import Transition
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplay

dataset_root = '/home/cvds_lab/Downloads/AgentHuman/'

train_set_root = os.path.join(dataset_root, 'SeqTrain')
validation_set_root = os.path.join(dataset_root, 'SeqVal')

# training set extraction
memory = ExperienceReplay(max_size=(MemoryGranularity.Transitions, sys.maxsize))
train_set_files = sorted(os.listdir(train_set_root))
print("found {} files".format(len(train_set_files)))
for file_idx, file in enumerate(train_set_files[:500]):
    print("extracting file {}: {}".format(file_idx, file))
    train_set = h5py.File(os.path.join(train_set_root, file), 'r')
    observations = train_set['rgb'][:]                                   # forward camera
    measurements = np.expand_dims(train_set['targets'][:, 10], -1)       # forward speed
    actions = train_set['targets'][:, :3]                                # steer, gas, brake
    actions[:, 1] -= actions[:, 2]
    actions = actions[:, :2][:, ::-1]

    high_level_commands = train_set['targets'][:, 24].astype('int') - 2  # follow lane, left, right, straight

    file_length = train_set['rgb'].len()
    assert train_set['rgb'].len() == train_set['targets'].len()

    for transition_idx in range(file_length):
        transition = Transition(
            state={
                'forward_camera': observations[transition_idx],
                'measurements': measurements[transition_idx],
                'high_level_command': high_level_commands[transition_idx]
            },
            action=actions[transition_idx],
            reward=0
        )
        memory.store(transition)

memory.save('carla_train_set_replay_buffer.p')
