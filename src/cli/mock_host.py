import argparse

import torch
from torch.utils.data import DataLoader
from data.MarkerTranslatorDataset import MarkerTranslatorDataset
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
from streaming.StreamingMocap import StreamingMocap
import threading
import re


class MockHostCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('mock-host', help='Test the streaming model against markers from a B3D file.')
        self.register_standard_options(subparser)
        subparser.add_argument('--b3d_path', type=str, help='Path to the B3D file to test.', default='../data/dev/Carter2023_Formatted_No_Arm_P003_split5.b3d')
        subparser.add_argument('--tsv-path', type=str, help='Path to the TSV file to test.', default='')
        subparser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        subparser.add_argument("--unscaled-generic-model", type=str, help="The path to the unscaled generic OpenSim model with the anatomical markerset.", default='../markerset.osim')
        subparser.add_argument("--geometry-path", type=str, help="The path to the Geometry/ folder.", default='../data/Geometry/')
        subparser.add_argument("--model-weights", type=str, help="The path to the model weights file.", default='../checkpoints/classifier_pretrained/classifier/epoch_2_batch_8230.pt')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'mock-host':
            return False

        b3d_path = os.path.abspath(args.b3d_path)
        tsv_path = args.tsv_path

        markers = []
        timestamps = []
        cop_torque_forces = []
        timestep = 0.01

        # Create an instance of the dataset
        if len(tsv_path) > 0:
            print('Loading markers from TSV file...')
            # Load the TSV file
            with open(tsv_path, 'r') as file:
                lines = file.readlines()
                for line in lines[1:]:
                    line_parts = re.split(r'[ \t]+', line.strip())
                    # timestamp = int(line_parts[0])
                    remaining_line_parts = line_parts[1:]
                    marker_obs: List[np.ndarray] = []
                    for i in range(0, len(remaining_line_parts) // 3):
                        marker_obs.append(np.array([float(remaining_line_parts[i*3]), float(remaining_line_parts[i*3+1]), float(remaining_line_parts[i*3+2])]))
                    markers.append(marker_obs)
                    timestamps.append(int(line_parts[0]) / 1000.0)
                    cop_torque_forces.append([])
            pass
        else:
            print('Loading markers from B3D file...')
            subject = nimble.biomechanics.SubjectOnDisk(os.path.abspath(b3d_path))
            trial = args.trial
            timestep = subject.getTrialTimestep(trial)
            frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                       0,
                                                                       subject.getTrialLength(trial),
                                                                       includeSensorData=True,
                                                                       includeProcessingPasses=False)
            for i, f in enumerate(frames):
                true_markers: List[Tuple[str, np.ndarray]] = f.markerObservations
                marker_obs: List[np.ndarray] = [pair[1] for pair in true_markers]
                marker_obs_cortex: List[np.ndarray] = []
                frame_cop_torque_forces: List[np.ndarray] = []
                for plate in range(len(frames[i].rawForcePlateForces)):
                    cop = frames[i].rawForcePlateCenterOfPressures[plate]
                    torque = frames[i].rawForcePlateTorques[plate]
                    forces = frames[i].rawForcePlateForces[plate]
                    cop_torque_force = np.expand_dims(np.array([cop[0] * 1000.0, cop[2] * 1000.0, cop[1] * 1000.0, torque[0], torque[2], torque[1], forces[0], forces[2], forces[1]]), axis=0)
                    frame_cop_torque_forces.append(cop_torque_force)
                cop_torque_forces.append(frame_cop_torque_forces)
                for marker in marker_obs:
                    marker_obs_cortex.append(np.array([marker[0], marker[2], marker[1]]) * 1000.0)
                markers.append(marker_obs_cortex)
                timestamps.append(i * subject.getTrialTimestep(trial))
        print('Loaded '+str(len(markers))+' timesteps from file.')

        # first_frame: nimble.biomechanics.Frame = \
        #     subject.readFrames(trial, 0, 1, includeSensorData=True, includeProcessingPasses=False)[0]
        # marker_names: List[str] = [pair[0] for pair in first_frame.markerObservations]
        # num_force_plates: int = len(first_frame.rawForcePlateForces)

        # Create the server
        server = nimble.biomechanics.CortexStreaming('127.0.0.1')

        marker_names = [str(i) for i in range(len(markers[i]))]
        marker_poses = markers[i]
        force_plate_cop_torque_forces = cop_torque_forces[0]
        server.mockServerSetData(marker_names, marker_poses, force_plate_cop_torque_forces)

        server.startMockServer()

        while True:
            for i in range(len(markers)):
                # Send the frame
                marker_names = [str(i) for i in range(len(markers[i]))]
                marker_poses = markers[i]
                force_plate_cop_torque_forces = cop_torque_forces[i]
                # print(marker_poses)
                server.mockServerSetData(marker_names, marker_poses, force_plate_cop_torque_forces)
                server.mockServerSendFrameMulticast()

                # Wait for the next frame
                time.sleep(timestep)
