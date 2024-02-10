import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np
from utils.TrainingMarkerLabel import TrainingMarkerLabel


class MarkerSupersetDataset(Dataset):
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
    skeleton_marker_name_to_index: Dict[TrainingMarkerLabel, int]
    skeleton_osim: str
    pad_with_random_unknown_markers: bool
    randomly_hide_markers_prob: float

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 output_class_tsv: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32,
                 testing_with_short_dataset: bool = False,
                 stride: int = 1,
                 output_data_format: str = 'last_frame',
                 skip_loading_skeletons: bool = True,
                 num_input_markers: int = 70,
                 randomly_hide_markers_prob: float = 0.3,
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
        self.overfit = overfit
        self.pad_with_random_unknown_markers = True
        self.randomly_hide_markers_prob = randomly_hide_markers_prob

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

        subject_file_names = [os.path.basename(subject_path) for subject_path in self.subject_paths]
        max_classification_index = 0
        with open(output_class_tsv, 'r') as f:
            lines = f.readlines()
            self.skeleton_marker_name_to_index = {}
            for line in lines[1:]:
                parts = line.strip().split('\t')
                subject_path = parts[0]
                classification_index = int(parts[2])
                subject_path_basename = os.path.basename(subject_path)
                if classification_index > max_classification_index:
                    max_classification_index = classification_index
                if subject_path_basename not in subject_file_names:
                    continue
                subject_index = subject_file_names.index(subject_path_basename)
                marker_name = parts[1]
                self.skeleton_marker_name_to_index[TrainingMarkerLabel(marker_name, subject_index)] = classification_index

        print(subject_file_names)

        self.max_input_markers = num_input_markers

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
                self.skeletons.append(skeleton)

            self.subjects.append(subject)
            # Prepare the list of windows we can use for training
            for trial_index in range(subject.getNumTrials()):
                trial_length = subject.getTrialLength(trial_index)
                for window_start in range(0, max(trial_length - self.window_size - 1, 0), self.window_size // 3):
                    assert window_start + self.window_size < trial_length
                    self.windows.append((i, trial_index, window_start))

        # Assign a unique index to the unknown marker
        self.unknown_marker_index = max_classification_index + 1

        print('Num classes: '+str(self.unknown_marker_index + 1))

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
            mask: torch.Tensor = torch.zeros(sequence_length, dtype=torch.int32)

            all_marker_names = []
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]
                for j in range(len(marker_obs)):
                    if marker_obs[j][0] not in all_marker_names:
                        all_marker_names.append(marker_obs[j][0])

            randomly_hide_marker: Dict[str, bool] = {}
            for marker_name in all_marker_names:
                randomly_hide_marker[marker_name] = torch.rand((1,)).item() < self.randomly_hide_markers_prob

            marker_obs_avg = np.zeros((3,))
            num_obs_markers = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]
                for j in range(len(marker_obs)):
                    if randomly_hide_marker[marker_obs[j][0]]:
                        continue
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    marker_obs_avg += marker_obs[j][1]
                    num_obs_markers += 1
            if num_obs_markers > 0:
                marker_obs_avg /= num_obs_markers

            assert not np.isnan(marker_obs_avg).any()

            rotation_about_y_axis = torch.rand((1,)).item() * 2 * np.pi
            rotation_matrix = np.array([[np.cos(rotation_about_y_axis), 0, np.sin(rotation_about_y_axis)],
                                        [0, 1, 0],
                                        [-np.sin(rotation_about_y_axis), 0, np.cos(rotation_about_y_axis)]])

            cursor = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]
                for j in range(min(len(marker_obs), self.max_input_markers)):
                    if randomly_hide_marker[marker_obs[j][0]]:
                        continue
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    input[cursor, :3] = torch.tensor(rotation_matrix @ (marker_obs[j][1] - marker_obs_avg), dtype=self.dtype)
                    input[cursor, 3] = float(i)
                    marker_label = TrainingMarkerLabel(marker_obs[j][0], subject_index)
                    if marker_label in self.skeleton_marker_name_to_index:
                        label[cursor] = self.skeleton_marker_name_to_index[marker_label]
                    else:
                        label[cursor] = self.unknown_marker_index
                    cursor += 1
            if cursor == 0:
                input[cursor, :3] = torch.randn(3, dtype=self.dtype)
                input[cursor, 3] = 0.0
                label[cursor] = self.unknown_marker_index
                cursor = 1

            if self.pad_with_random_unknown_markers:
                # First, add some static irrelevant markers, since that's a common case
                available_pad = sequence_length - cursor
                if available_pad > 0:
                    random_pad_stillness = torch.randint(0, available_pad // len(frames), (1,)).item()
                    for i in range(random_pad_stillness):
                        point = torch.randn(3, dtype=self.dtype)
                        for t in range(len(frames)):
                            input[cursor, :3] = point + torch.randn(3, dtype=self.dtype) * 0.005
                            input[cursor, 3] = t
                            label[cursor] = self.unknown_marker_index
                            cursor += 1
                # Next, add some random noise irrelevant markers, since that's the other common case
                available_pad = sequence_length - cursor
                if available_pad > 0:
                    random_pad = torch.randint(0, available_pad, (1,)).item()
                    for i in range(random_pad):
                        input[cursor, :3] = torch.randn(3, dtype=self.dtype)
                        input[cursor, 3] = torch.randint(0, len(frames), (1,)).float()
                        label[cursor] = self.unknown_marker_index
                        cursor += 1
            mask[:cursor] = 1

        # Assert there are no NaNs in input, label, or mask
        assert not torch.any(torch.isnan(input))
        assert not torch.any(torch.isnan(label))
        assert not torch.any(torch.isnan(mask))

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
            print('Loading subject header ' + str(i + 1) + '/' + str(len(self.subject_paths)))
            self.subjects.append(subject)
        print('Done!')


