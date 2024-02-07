import os.path
import unittest
import torch
import torch.optim as optim
import numpy as np
import nimblephysics as nimble
from typing import List
from src.loss.MaskedCrossEntropyLoss import MaskedCrossEntropyLoss


class TestMaskedCrossEntropyLoss(unittest.TestCase):
    @staticmethod
    def featurize_truly_original(subject: nimble.biomechanics.SubjectOnDisk,
                                 trial: int = 0,
                                 start: int = 0,
                                 window_size: int = 10,
                                 stride: int = 5,
                                 max_input_markers: int = 50):
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   start,
                                                                   window_size // stride,
                                                                   stride=stride,
                                                                   includeSensorData=True,
                                                                   includeProcessingPasses=False)
        assert (len(frames) == window_size // stride)

        sequence_length = (window_size // stride) * max_input_markers

        with torch.no_grad():
            input: torch.Tensor = torch.zeros((sequence_length, 4), dtype=torch.float32)
            mask: torch.Tensor = torch.zeros(sequence_length, dtype=torch.float32)

            marker_obs_avg = np.zeros((3,))
            num_obs_markers = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:max_input_markers]
                for j in range(len(marker_obs)):
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    marker_obs_avg += marker_obs[j][1]
                    num_obs_markers += 1
            if num_obs_markers > 0:
                marker_obs_avg /= num_obs_markers

            assert not np.isnan(marker_obs_avg).any()

            cursor = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:max_input_markers]
                for j in range(min(len(marker_obs), max_input_markers)):
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    input[cursor, :3] = torch.tensor(marker_obs[j][1] - marker_obs_avg, dtype=self.dtype)
                    input[cursor, 3] = float(i)
                    cursor += 1
            if cursor == 0:
                input[cursor, :3] = torch.randn(3, dtype=self.dtype)
                input[cursor, 3] = 0.0
                cursor = 1
            mask[:cursor] = 1

        # Assert there are no NaNs in input, label, or mask
        assert not torch.any(torch.isnan(input))
        assert not torch.any(torch.isnan(mask))

        return input, mask

    @staticmethod
    def featurize_no_extra_padding(subject: nimble.biomechanics.SubjectOnDisk,
                                   trial: int = 0,
                                   start: int = 0,
                                   window_size: int = 10,
                                   stride: int = 5,
                                   max_input_markers: int = 50):
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   start,
                                                                   window_size // stride,
                                                                   stride=stride,
                                                                   includeSensorData=True,
                                                                   includeProcessingPasses=False)
        assert (len(frames) == window_size // stride)

        input_vectors: List[np.ndarray] = []

        with torch.no_grad():
            marker_obs_avg = np.zeros((3,))
            num_obs_markers = 0
            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:max_input_markers]
                for j in range(len(marker_obs)):
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    marker_obs_avg += marker_obs[j][1]
                    num_obs_markers += 1
            if num_obs_markers > 0:
                marker_obs_avg /= num_obs_markers

            assert not np.isnan(marker_obs_avg).any()

            for i in range(len(frames)):
                marker_obs = frames[i].markerObservations[:max_input_markers]
                for j in range(min(len(marker_obs), max_input_markers)):
                    # if marker_obs has any nan, skip it
                    if np.isnan(marker_obs[j][1]).any():
                        continue
                    input_vec = np.zeros(4, dtype=np.float32)
                    input_vec[:3] = marker_obs[j][1] - marker_obs_avg
                    input_vec[3] = float(i)
                    input_vectors.append(input_vec)

        return np.stack(input_vectors)

    @staticmethod
    def featurize_native_accelerated(subject: nimble.biomechanics.SubjectOnDisk,
                                     trial: int = 0,
                                     start: int = 0,
                                     window_size: int = 10,
                                     stride: int = 5,
                                     max_input_markers: int = 50):
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   start,
                                                                   window_size,
                                                                   stride=1,
                                                                   includeSensorData=True,
                                                                   includeProcessingPasses=False)
        assert (len(frames) == window_size)
        streaming: nimble.biomechanics.StreamingMarkerTraces = nimble.biomechanics.StreamingMarkerTraces(50, window_size // stride, stride, max_input_markers)

        for i in range(len(frames)):
            marker_obs = frames[i].markerObservations[:max_input_markers]
            raw_marker_points: List[np.ndarray] = []
            for j in range(len(marker_obs)):
                # if marker_obs has any nan, skip it
                if np.isnan(marker_obs[j][1]).any():
                    continue
                raw_marker_points.append(marker_obs[j][1])
            streaming.observeMarkers(raw_marker_points, i * 10)

        features, logits = streaming.getTraceFeatures(center=True)
        return features.transpose()

    def test_native_featurizing_equality(self):
        subject_path = os.path.abspath('../data/Falisse_subject_1_raw.b3d')
        subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(subject_path)

        windows = 6
        stride = 3

        feat1 = self.featurize_no_extra_padding(subject, window_size=windows, stride=stride)
        feat2 = self.featurize_native_accelerated(subject, window_size=windows, stride=stride)

        # Check confusion matrix
        np.testing.assert_allclose(feat1, feat2)


if __name__ == '__main__':
    unittest.main()
