import argparse

import nimblephysics_libs.biomechanics

from cli.visualize import VisualizeCommand
from cli.train import TrainCommand
from cli.create_marker_superset import CreateMarkerSuperset
from cli.mock_stream import MockStreamCommand
from cli.real_stream import RealStreamCommand
from cli.record import RecordCommand
from cli.eval import EvalCommand
from cli.mock_host import MockHostCommand
import nimblephysics as nimble
import logging

def main():
    commands = [VisualizeCommand(), TrainCommand(), CreateMarkerSuperset(), MockStreamCommand(), EvalCommand(), RealStreamCommand(), RecordCommand(), MockHostCommand()]

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='MarkerTranslator Command Line Interface')

    # Split up by command
    subparsers = parser.add_subparsers(dest="command")

    # Add a parser for each command
    for command in commands:
        command.register_subcommand(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    for command in commands:
        if command.run(args):
            return


if __name__ == '__main__':
    logpath = "log"
    # Create and configure logger
    logging.basicConfig(filename=logpath,
                        format='%(asctime)s %(message)s',
                        filemode='a')

    # Creating an object
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    # Setting the threshold of logger to INFO
    logger.setLevel(logging.INFO)
    main()