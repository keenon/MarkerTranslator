import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class MarkerLabelerDataset(Dataset):
    stride: int
    data_path: str
    window_size: int
    geometry_folder: str
    device: torch.device
    dtype: torch.dtype
    subject_paths: List[str]
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    overfit: bool
    windows: List[Tuple[int, int, int]]  # Subject, trial, start_frame
    skeletons: List[nimble.dynamics.Skeleton]
    max_input_markers: int
    marker_name_to_body_index: Dict[str, int]

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32,
                 testing_with_short_dataset: bool = False,
                 stride: int = 1,
                 output_data_format: str = 'last_frame',
                 skip_loading_skeletons: bool = False,
                 num_input_markers: int = 70,
                 overfit: bool = False):
        self.stride = stride
        self.output_data_format = output_data_format
        self.subject_paths = []
        self.subjects = []
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.dtype = dtype
        self.windows = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_markersets = []
        self.overfit = overfit

        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d") and "vander" not in file.lower():
                        self.subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".b3d")
            self.subject_paths.append(data_path)

        if testing_with_short_dataset:
            self.subject_paths = self.subject_paths[:5]
        if overfit:
            self.subject_paths = self.subject_paths[:1]

        self.max_input_markers = num_input_markers
        self.marker_name_to_body_index = {}
        self.unknown_marker_index = 0

        # Walk the folder path, and check for any with the ".b3d" extension (indicating that they are
        # AddBiomechanics binary data files)
        for i, subject_path in enumerate(self.subject_paths):
            # Add the skeleton to the list of skeletons
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            if not skip_loading_skeletons:
                print('Loading skeleton ' + str(i + 1) + '/' + str(
                    len(self.subject_paths)) + f' for subject {subject_path}')
                osim = subject.readOpenSimFile(subject.getNumProcessingPasses() - 1, geometry_folder)
                skeleton = osim.skeleton
                self.unknown_marker_index = skeleton.getNumBodyNodes()
                for marker in osim.markersMap:
                    self.marker_name_to_body_index[marker] = osim.markersMap[marker][0].getIndexInSkeleton()
                self.skeletons.append(skeleton)

            self.subjects.append(subject)
            # Prepare the list of windows we can use for training
            for trial_index in range(subject.getNumTrials()):
                trial_length = subject.getTrialLength(trial_index)
                for window_start in range(max(trial_length - self.window_size - 1, 0)):
                    assert window_start + self.window_size < trial_length
                    self.windows.append((i, trial_index, window_start))

        if overfit:
            self.windows = self.windows[:256]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        subject = self.subjects[subject_index]
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   self.window_size // self.stride,
                                                                   stride=self.stride,
                                                                   includeSensorData=True,
                                                                   includeProcessingPasses=False)
        assert (len(frames) == self.window_size // self.stride)

        sequence_length = (self.window_size // self.stride) * self.max_input_markers

        with torch.no_grad():
            input: torch.Tensor = torch.zeros((sequence_length, 4), dtype=self.dtype)
            label: torch.Tensor = torch.zeros(sequence_length, dtype=torch.int64)
            mask: torch.Tensor = torch.zeros(sequence_length, dtype=self.dtype)

            marker_obs_avg = np.zeros((3,))
            num_obs_markers = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]
                for j in range(len(marker_obs)):
                    marker_obs_avg += marker_obs[j][1]
                    num_obs_markers += 1
            if num_obs_markers > 0:
                marker_obs_avg /= num_obs_markers

            cursor = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]
                for j in range(min(len(marker_obs), self.max_input_markers)):
                    input[cursor, :3] = torch.tensor(marker_obs[j][1] - marker_obs_avg, dtype=self.dtype)
                    input[cursor, 3] = i
                    if marker_obs[j][0] in self.marker_name_to_body_index:
                        label[cursor] = self.marker_name_to_body_index[marker_obs[j][0]]
                    else:
                        label[cursor] = self.unknown_marker_index
                    cursor += 1
            mask[:cursor] = 1

        return input, label, mask, subject_index, trial

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['subjects']
        del state['skeletons']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        self.subjects = []
        self.skeletons = []
        print('Unpickling AddBiomechanicsDataset copy in reader worker thread')
        # Create the non picklable SubjectOnDisk objects.
        for i, subject_path in enumerate(self.subject_paths):
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            self.subjects.append(subject)


