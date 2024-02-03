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


class RecordCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('record', help='Record data coming from Cortex to a file.')
        self.register_standard_options(subparser)
        subparser.add_argument('--cortex-host', type=str, help='The IP of the Cortex SDK. Defaults to 127.0.0.1',
                            default='127.0.0.1')
        subparser.add_argument('--output-path', type=str, help='The path to the output file.', default='')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'record':
            return False

        cortex_host: str = args.cortex_host
        output_path: str = args.output_path

        client = nimble.biomechanics.CortexStreaming(cortex_host)

        gui = NimbleGUI()
        gui.serve(8080)

        # Open the file
        if output_path != '':
            with open(output_path, 'w') as f:
                f.write('time\t')
                f.write(' '.join([f'marker_{i}_x marker_{i}_y marker_{i}_z' for i in range(100)]))
                f.write('\n')

                def on_frame(marker_names: List[str], marker_poses: List[np.ndarray], force_plate_cop_torque_forces: List[np.ndarray]):
                    current_time_milliseconds = int(round(time.time() * 1000))
                    f.write(f'{current_time_milliseconds}\t')
                    for i, pos in enumerate(marker_poses):
                        f.write(f'{pos[0]} {pos[1]} {pos[2]} ')
                    f.write('\n')

                    gui.nativeAPI().clear()
                    for i, pos in enumerate(marker_poses):
                        gui.nativeAPI().createBox(str(i), np.ones(3) * 0.05, pos * 0.001, np.zeros(3), [0.5, 0.5, 0.5, 1.0])

                client.setFrameHandler(on_frame)
                client.initialize()

                gui.blockWhileServing()
        else:
            print('No output file specified. Just rendering markers to the GUI')

            def on_frame(marker_names: List[str], marker_poses: List[np.ndarray], force_plate_cop_torque_forces: List[np.ndarray]):
                gui.nativeAPI().clear()
                print(len(marker_poses))
                for i, pos in enumerate(marker_poses):
                    gui.nativeAPI().createBox(str(i), np.ones(3) * 0.05, pos * 0.001, np.zeros(3), [0.5, 0.5, 0.5, 1.0])

            client.setFrameHandler(on_frame)
            client.initialize()

            gui.blockWhileServing()
