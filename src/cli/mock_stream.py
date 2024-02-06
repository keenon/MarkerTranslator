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


class MockStreamCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('mock-stream', help='Test the streaming model against markers from a B3D file.')
        self.register_standard_options(subparser)
        subparser.add_argument('--b3d_path', type=str, help='Path to the B3D file to test.', default='../data/dev/Carter2023_Formatted_No_Arm_P003_split5.b3d')
        subparser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        subparser.add_argument("--unscaled-generic-model", type=str, help="The path to the unscaled generic OpenSim model with the anatomical markerset.", default='../markerset.osim')
        subparser.add_argument("--geometry-path", type=str, help="The path to the Geometry/ folder.", default='../data/Geometry/')
        subparser.add_argument("--model-weights", type=str, help="The path to the model weights file.", default='../checkpoints/classifier_pretrained/classifier/epoch_2_batch_8230.pt')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'mock-stream':
            return False

        b3d_path = os.path.abspath(args.b3d_path)
        unscaled_generic_model_path = args.unscaled_generic_model
        weights_path = args.model_weights
        geometry_path = args.geometry_path
        transformer_dim: int = args.transformer_dim
        transformer_nheads: int = args.transformer_nheads
        transformer_nlayers: int = args.transformer_nlayers

        streaming = StreamingMocap(unscaled_generic_model_path, geometry_path, weights_path, d_model=transformer_dim, nhead=transformer_nheads, num_transformer_layers=transformer_nlayers, dim_feedforward=transformer_dim)
        streaming.start_gui()
        streaming.start_inference_process()
        streaming.start_ik_thread()

        # Create an instance of the dataset
        subject = nimble.biomechanics.SubjectOnDisk(os.path.abspath(b3d_path))

        trial: int = 0
        frame: int = 0
        playing: bool = True

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(subject.getTrialTimestep(0))

        def inference_thread():
            nonlocal streaming
            nonlocal playing

            while True:
                if playing:
                    now_ms = int(round(time.time() * 1000))
                    streaming.run_model(now_ms)
                    time.sleep(0.5)

        inference_thread = threading.Thread(target=inference_thread)
        inference_thread.start()

        def on_tick(now_ms: int):
            nonlocal frame
            nonlocal trial
            nonlocal playing

            frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                       frame,
                                                                       1,
                                                                       includeSensorData=True,
                                                                       includeProcessingPasses=False)

            true_markers: List[Tuple[str, np.ndarray]] = frames[0].markerObservations
            marker_obs: List[np.ndarray] = [pair[1] for pair in true_markers]

            streaming.observe_markers(marker_obs, now_ms / 1000.0)

            if playing:
                frame += 1
                if frame >= subject.getTrialLength(trial):
                    streaming.reset()
                    frame = 0
                    trial += 1
                    print("Trial {} complete.".format(trial))
                    if trial >= subject.getNumTrials():
                        print("All {} trials complete.".format(subject.getNumTrials()))
                        trial = 0

        ticker.registerTickListener(on_tick)
        ticker.start()

        # time_ms = 0
        # timestep_millis = int(subject.getTrialTimestep(0) * 1000)
        # while True:
        #     on_tick(time_ms)
        #     time_ms += timestep_millis
        #     time.sleep(timestep_millis / 1000.0)

        streaming.gui.blockWhileServing()

