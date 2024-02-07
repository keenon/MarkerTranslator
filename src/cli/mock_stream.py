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


class MockStreamCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('mock-stream', help='Test the streaming model against markers from a B3D file.')
        self.register_standard_options(subparser)
        subparser.add_argument('--b3d_path', type=str, help='Path to the B3D file to test.', default='../data/dev/Carter2023_Formatted_No_Arm_P003_split5.b3d')
        subparser.add_argument('--tsv-path', type=str, help='Path to the TSV file to test.', default='')
        subparser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        subparser.add_argument("--unscaled-generic-model", type=str, help="The path to the unscaled generic OpenSim model with the anatomical markerset.", default='../markerset.osim')
        subparser.add_argument("--geometry-path", type=str, help="The path to the Geometry/ folder.", default='../data/Geometry/')
        subparser.add_argument("--model-weights", type=str, help="The path to the model weights file.", default='../checkpoints/classifier_pretrained/classifier/epoch_2_batch_8230.pt')
        subparser.add_argument("--anthro-xml", type=str, help="The path to the anthropometrics XML file.", default='../data/ANSUR_metrics.xml')
        subparser.add_argument("--anthro-data", type=str, help="The path to the anthropometrics data file.", default='../data/ANSUR_II_BOTH_Public.csv')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'mock-stream':
            return False

        b3d_path = os.path.abspath(args.b3d_path)
        tsv_path = args.tsv_path
        unscaled_generic_model_path = args.unscaled_generic_model
        weights_path = args.model_weights
        geometry_path = args.geometry_path
        transformer_dim: int = args.transformer_dim
        transformer_nheads: int = args.transformer_nheads
        transformer_nlayers: int = args.transformer_nlayers
        anthro_xml: str = os.path.abspath(args.anthro_xml)
        anthro_data: str = os.path.abspath(args.anthro_data)

        streaming = StreamingMocap(unscaled_generic_model_path, geometry_path, weights_path, d_model=transformer_dim, nhead=transformer_nheads, num_transformer_layers=transformer_nlayers, dim_feedforward=transformer_dim)
        streaming.set_anthropometrics(anthro_xml, anthro_data)
        streaming.start_gui()
        streaming.start_inference_process()
        streaming.start_ik_thread()

        markers = []
        timestamps = []
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
                        marker_obs.append(np.array([float(remaining_line_parts[i*3]), float(remaining_line_parts[i*3+2]), float(remaining_line_parts[i*3+1])]) * 0.001)
                    markers.append(marker_obs)
                    timestamps.append(int(line_parts[0]) / 1000.0)
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
                markers.append(marker_obs)
                timestamps.append(i * subject.getTrialTimestep(trial))
        print('Loaded '+str(len(markers))+' timesteps from file.')

        frame: int = 0
        playing: bool = True

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(timestep)

        def inference_thread():
            nonlocal streaming
            nonlocal playing

            while True:
                if playing:
                    streaming.run_model()
                    time.sleep(0.5)

        inference_thread = threading.Thread(target=inference_thread)
        inference_thread.start()

        def on_tick(now_ms: int):
            nonlocal frame
            nonlocal playing
            nonlocal markers

            streaming.observe_markers(markers[frame], timestamps[frame])

            if playing:
                frame += 1
                if frame >= len(markers):
                    streaming.reset()
                    frame = 0
                    print('Resetting')

        ticker.registerTickListener(on_tick)
        if len(markers) > 0:
            ticker.start()

        # time_ms = 0
        # timestep_millis = int(subject.getTrialTimestep(0) * 1000)
        # while True:
        #     on_tick(time_ms)
        #     time_ms += timestep_millis
        #     time.sleep(timestep_millis / 1000.0)

        streaming.gui.blockWhileServing()

