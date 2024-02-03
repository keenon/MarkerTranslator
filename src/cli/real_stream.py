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


class RealStreamCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('stream', help='Test the streaming model against real Cortex data.')
        self.register_standard_options(subparser)
        subparser.add_argument('--cortex-host', type=str, help='The IP of the Cortex SDK. Defaults to 127.0.0.1',
                            default='127.0.0.1')
        subparser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        subparser.add_argument("--unscaled-generic-model", type=str, help="The path to the unscaled generic OpenSim model with the anatomical markerset.", default='../markerset.osim')
        subparser.add_argument("--geometry-path", type=str, help="The path to the Geometry/ folder.", default='../data/Geometry/')
        subparser.add_argument("--model-weights", type=str, help="The path to the model weights file.", default='../checkpoints/classifier_pretrained/classifier/epoch_2_batch_8230.pt')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'stream':
            return False

        unscaled_generic_model_path = args.unscaled_generic_model
        weights_path = args.model_weights
        geometry_path = args.geometry_path
        transformer_dim: int = args.transformer_dim
        transformer_nheads: int = args.transformer_nheads
        transformer_nlayers: int = args.transformer_nlayers
        cortex_host: str = args.cortex_host

        client = nimble.biomechanics.CortexStreaming(cortex_host)

        streaming = StreamingMocap(unscaled_generic_model_path, geometry_path, weights_path, d_model=transformer_dim, nhead=transformer_nheads, num_transformer_layers=transformer_nlayers, dim_feedforward=transformer_dim)
        streaming.start_gui()
        streaming.start_inference_process()

        playing: bool = True

        def inference_thread():
            nonlocal streaming
            nonlocal playing

            while True:
                if playing:
                    streaming.run_model()
                    time.sleep(0.5)

        inference_thread = threading.Thread(target=inference_thread)
        inference_thread.start()

        def on_frame(marker_names: List[str], marker_poses: List[np.ndarray], force_plate_cop_torque_forces: List[np.ndarray]):
            current_time_milliseconds = int(round(time.time() * 1000))
            translated_marker_poses: List[np.ndarray] = []
            for pos in marker_poses:
                translated_pos = np.array([pos[0] * 0.001, pos[2] * 0.001, pos[1] * 0.001])
                translated_marker_poses.append(translated_pos)
            time_sec = current_time_milliseconds / 1000.0
            streaming.observe_markers(translated_marker_poses, time_sec)
            streaming.run_ik_update()

        client.setFrameHandler(on_frame)
        client.initialize()

        # Don't exit until the user presses Ctrl+C
        streaming.gui.blockWhileServing()

