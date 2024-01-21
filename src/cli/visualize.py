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


class VisualizeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('visualize', help='Visualize the performance of a model on dataset.')
        self.register_standard_options(subparser)
        subparser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        subparser.add_argument("--playback-speed", type=float, help="The playback speed for the GUI.", default=0.5)

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'visualize':
            return False

        # Create an instance of the dataset
        print('## Loading set:')
        dataset = self.get_dataset(args, 'train')

        # Create an instance of the model
        model = self.get_model(args)
        self.load_latest_checkpoint(model)

        gui = NimbleGUI()
        gui.serve(8080)

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
            0.04)

        frame: int = 0
        playing: bool = True
        num_frames = len(dataset)
        if num_frames == 0:
            print('No frames in dataset!')
            exit(1)

        def onKeyPress(key):
            nonlocal playing
            nonlocal frame
            if key == ' ':
                playing = not playing
            elif key == 'e':
                frame += 1
                if frame >= num_frames - 5:
                    frame = 0
            elif key == 'a':
                frame -= 1
                if frame < 0:
                    frame = num_frames - 5

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame
                nonlocal dataset

                print('Frame: ' + str(frame) + ' / ' + str(num_frames))

                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                inputs, labels, batch_subject_index, trial_index = dataset[frame]
                batch_subject_indices: List[int] = [batch_subject_index]
                batch_trial_indices: List[int] = [trial_index]

                # Add a batch dimension
                inputs = inputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
                outputs = model(inputs)

                # Visualize the input markers
                input_markers = inputs[0, 0, :]
                for i in range(0, len(input_markers), 3):
                    pos = input_markers[i:i + 3].detach().numpy()
                    gui.nativeAPI().createBox('input_marker_' + str(i), 0.05 * np.ones(3), pos, np.zeros(3), np.array([1.0, 0.5, 0.5, 1.0]))

                # Visualize the output and label markers
                output_markers = outputs[0, 0, :]
                label_markers = labels[0, 0, :]
                for i in range(0, len(label_markers), 3):
                    label_pos = label_markers[i:i + 3].detach().numpy()
                    output_pos = output_markers[i:i + 3].detach().numpy()
                    gui.nativeAPI().createBox('label_marker_' + str(i), 0.05 * np.ones(3), label_pos, np.zeros(3), np.array([0.5, 0.5, 1.0, 1.0]))
                    gui.nativeAPI().createBox('output_marker_' + str(i), 0.05 * np.ones(3), output_pos, np.zeros(3), np.array([0.5, 1.0, 0.5, 1.0]))
                    gui.nativeAPI().createLine('line_' + str(i), [label_pos, output_pos], np.array([0.5, 0.5, 1.0, 1.0]))

                if playing:
                    frame += 1
                    if frame >= num_frames - 5:
                        frame = 0

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True

