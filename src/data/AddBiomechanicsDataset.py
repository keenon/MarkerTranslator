import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class AddBiomechanicsDataset(Dataset):
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
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    marker_set: List[Tuple[str, np.ndarray]]
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_markersets: List[List[Tuple[nimble.dynamics.BodyNode, np.ndarray]]]
    max_input_markers: int

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
                 num_output_markers: int = 30,
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
                for window_start in range(max(trial_length - self.window_size - 1, 0)):
                    assert window_start + self.window_size < trial_length
                    self.windows.append((i, trial_index, window_start))

        # Load the marker set
        self.marker_set = []
        if len(self.skeletons) > 0:
            skel = self.skeletons[0]
            for b in range(skel.getNumBodyNodes()):
                body = skel.getBodyNode(b)
                name: str = body.getName()
                self.marker_set.append((name, np.array([0.05, 0., 0.])))
                self.marker_set.append((name, np.array([0., 0.05, 0.])))
                self.marker_set.append((name, np.array([0., 0., 0.05])))
        print('Marker set: ' + str(len(self.marker_set)))
        assert(len(self.marker_set) == num_output_markers)
        for skeleton in self.skeletons:
            markerset = [(skeleton.getBodyNode(name), offset) for name, offset in self.marker_set]
            for body, offset in markerset:
                assert body is not None
            self.skeletons_markersets.append(markerset)

        if overfit:
            self.windows = self.windows[:200]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        subject = self.subjects[subject_index]
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   self.window_size // self.stride,
                                                                   stride=self.stride,
                                                                   includeSensorData=True,
                                                                   includeProcessingPasses=True)
        assert (len(frames) == self.window_size // self.stride)
        pos_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[-1] for frame in frames]

        with torch.no_grad():
            input: torch.Tensor = torch.zeros((self.window_size // self.stride, self.max_input_markers * 3), dtype=self.dtype)

            label: torch.Tensor = torch.zeros((self.window_size // self.stride, len(self.marker_set) * 3), dtype=self.dtype)
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:self.max_input_markers]

                marker_obs_avg = np.zeros((3,))
                for j in range(len(marker_obs)):
                    marker_obs_avg += marker_obs[j][1]
                if len(marker_obs) > 0:
                    marker_obs_avg /= len(marker_obs)

                for j in range(min(len(marker_obs), self.max_input_markers)):
                    input[i, j * 3:j * 3 + 3] = torch.tensor(marker_obs[j][1] - marker_obs_avg, dtype=self.dtype)
                self.skeletons[subject_index].setPositions(pos_passes[i].pos)
                marker_world_positions: np.ndarray = self.skeletons[subject_index].getMarkerWorldPositions(self.skeletons_markersets[subject_index])
                for j in range(len(marker_world_positions) // 3):
                    label[i, j * 3:j * 3 + 3] = torch.tensor(marker_world_positions[j*3: j*3 + 3] - marker_obs_avg, dtype=self.dtype)

        return input, label, subject_index, trial

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['subjects']
        del state['skeletons']
        del state['skeletons_markersets']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        self.subjects = []
        self.skeletons = []
        self.skeletons_markersets = []
        print('Unpickling AddBiomechanicsDataset copy in reader worker thread')
        # Create the non picklable SubjectOnDisk objects.
        for i, subject_path in enumerate(self.subject_paths):
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            self.subjects.append(subject)
            self.skeletons.append(subject.readSkel(0, ignoreGeometry=True))
        for skeleton in self.skeletons:
            markerset = [(skeleton.getBodyNode(name), offset) for name, offset in self.marker_set]
            for body, offset in markerset:
                assert body is not None
            self.skeletons_markersets.append(markerset)


