import argparse
import os
import torch
from data.MarkerTranslatorDataset import MarkerTranslatorDataset
from data.MarkerLabelerDataset import MarkerLabelerDataset
from data.MarkerSupersetDataset import MarkerSupersetDataset
from models.MaxPoolLSTMPointCloudRegressor import MaxPoolLSTMPointCloudRegressor
from models.TransformerSequenceClassifier import TransformerSequenceClassifier
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from loss.MaskedCrossEntropyLoss import MaskedCrossEntropyLoss
from typing import List
import logging


class AbstractCommand:
    """
    All of our different activities inherit from this class. This class defines the interface for a CLI command, so
    that it's convenient to split commands across files. It also carries shared logic for loading / saving models, etc.
    """

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        pass

    def run(self, args: argparse.Namespace) -> bool:
        pass

    def register_standard_options(self, subparser: argparse.ArgumentParser):
        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='classifier',
                               help='The model to type to train. Options are "classifier" and "regressor".')
        subparser.add_argument('--output-data-format', type=str, default='all_frames',
                               choices=['all_frames', 'last_frame'],
                               help='Output for all frames in a window or only the last frame.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                               help='The path to a model checkpoint to save during training. Also, starts from the '
                                    'latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None,
                               help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=150,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=15,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--num-input-markers', type=int, default=70,
                               help='The (maximum) number of input markers the model can take.')
        subparser.add_argument('--num-output-markers', type=int, default=36,
                               help='The number of output markers the model produces.')
        subparser.add_argument('--output-class-tsv', type=str, default='../class_map.tsv',
                               help='The map from file and marker name to class number.')
        subparser.add_argument('--dropout', action='store_true', default=True, help='Apply dropout?')
        subparser.add_argument('--dropout-prob', type=float, default=0.3, help='Dropout prob')
        subparser.add_argument('--in-hidden-dims', type=int, nargs='+', default=[128],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--time-hidden-dim', type=int, nargs='+', default=1024,
                               help='Size of the hidden dimension across time, summarizing a timestep.')
        subparser.add_argument('--out-hidden-dims', type=int, nargs='+', default=[1024],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='sigmoid', help='Which activation func?')
        subparser.add_argument('--short', action='store_true',
                               help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--overfit', action='store_true',
                               help='Use a tiny dataset to check if the model can overfit the data.')
        pass

    def ensure_geometry(self, geometry: str):
        if geometry is None:
            # Check if the "./Geometry" folder exists, and if not, download it
            if not os.path.exists('./Geometry'):
                print('Downloading the Geometry folder from https://addbiomechanics.org/resources/Geometry.zip')
                exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
                if exit_code != 0:
                    print('ERROR: Failed to download Geometry.zip. You may need to install wget. If you are on a Mac, '
                          'try running "brew install wget"')
                    return False
                os.system('unzip ./Geometry.zip')
                os.system('rm ./Geometry.zip')
            geometry = './Geometry'
        print('Using Geometry folder: ' + geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
        return geometry

    def get_num_classes(self, args):
        max_classification_index = 0
        output_class_tsv: str = args.output_class_tsv
        with open(output_class_tsv, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split('\t')
                classification_index = int(parts[2])
                if classification_index > max_classification_index:
                    max_classification_index = classification_index
        num_classes = max_classification_index + 2
        return num_classes

    def get_model(self, args):
        model_type: str = args.model_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), model_type)
        history_len: int = args.history_len
        in_hidden_dims: List[int] = args.in_hidden_dims
        out_hidden_dims: List[int] = args.out_hidden_dims
        device: str = args.device
        dtype: torch.dtype = torch.float64
        stride: int = args.stride
        batchnorm: bool = args.batchnorm
        dropout: bool = args.dropout
        dropout_prob: float = args.dropout_prob
        activation: str = args.activation
        num_input_markers: int = args.num_input_markers
        num_output_markers: int = args.num_output_markers
        time_hidden_dim: int = args.time_hidden_dim


        if model_type == "regressor":
            return MaxPoolLSTMPointCloudRegressor(history_len=history_len,
                                                  num_input_markers=num_input_markers,
                                                  num_output_markers=num_output_markers,
                                                  in_mlp_hidden_dims=in_hidden_dims,
                                                  time_hidden_dim=time_hidden_dim,
                                                  out_mlp_hidden_dims=out_hidden_dims,
                                                  dropout=dropout,
                                                  dropout_prob=dropout_prob,
                                                  batchnorm=batchnorm,
                                                  activation=activation,
                                                  device=device)
        else:
            assert (model_type == "classifier")
            num_classes = self.get_num_classes(args)
            return TransformerSequenceClassifier(num_classes=num_classes, device=device)

    def get_dataset(self, args, suffix: str):
        model_type: str = args.model_type
        dataset_home: str = args.dataset_home
        history_len: int = args.history_len
        device: str = args.device
        short: bool = args.short
        stride: int = args.stride
        geometry = self.ensure_geometry(args.geometry_folder)
        num_input_markers: int = args.num_input_markers
        num_output_markers: int = args.num_output_markers
        overfit: bool = args.overfit
        output_class_tsv: str = args.output_class_tsv

        dataset_path = os.path.abspath(os.path.join(dataset_home, suffix))
        if model_type == "regressor":
            dataset = MarkerTranslatorDataset(dataset_path,
                                              history_len,
                                              num_input_markers=num_input_markers,
                                              num_output_markers=num_output_markers,
                                              device=torch.device(device),
                                              stride=stride,
                                              geometry_folder=geometry,
                                              testing_with_short_dataset=short,
                                              overfit=overfit)
        else:
            assert (model_type == "classifier")
            dataset = MarkerSupersetDataset(dataset_path,
                                            history_len,
                                            geometry_folder=geometry,
                                            output_class_tsv=output_class_tsv,
                                            num_input_markers=num_input_markers,
                                            device=torch.device(device),
                                            stride=stride,
                                            testing_with_short_dataset=short,
                                            overfit=overfit)
        return dataset

    def get_loss(self, args, split: str):
        model_type: str = args.model_type
        num_classes = self.get_num_classes(args)
        if model_type == "regressor":
            return RegressionLossEvaluator(split, num_classes)
        else:
            assert (model_type == "classifier")
            return MaskedCrossEntropyLoss(split, num_classes)

    def load_latest_checkpoint(self, model, optimizer=None, checkpoint_dir="../checkpoints/lstm"):
        if not os.path.exists(checkpoint_dir):
            print("Checkpoint directory does not exist!")
            return

        # Get all the checkpoint files
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

        # If there are no checkpoints, return
        if not checkpoints:
            print("No checkpoints available!")
            return

        # Sort the files based on the epoch and batch numbers in their filenames
        checkpoints.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))

        # Get the path of the latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

        logging.info(f"{latest_checkpoint=}")
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint)

        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # You might also want to return the epoch and batch number so you can continue training from there
        epoch = checkpoint['epoch']
        batch = checkpoints[-1].split('_')[3].split('.')[0]

        print(f"Loaded checkpoint from epoch {epoch}, batch {batch}")

        return epoch, int(batch)
