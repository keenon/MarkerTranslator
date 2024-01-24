import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging
import nimblephysics as nimble
from utils.TrainingMarkerLabel import TrainingMarkerLabel



class UnifiedAnatomicalMarker:
    # Which body node is this marker attached to?
    body: str

    # Where is the marker on that body?
    offset: np.ndarray

    # What is this marker called, in each skeleton that it is included in in the training set?
    labels: List[TrainingMarkerLabel]

    # What is the index of this marker in the unified marker set?
    classification_index: int

    merged_offsets: List[np.ndarray]

    def __init__(self, label: TrainingMarkerLabel, body: str, offset: np.ndarray):
        self.labels = [label]
        self.body = body
        self.offset = offset
        self.classification_index = -1
        self.merged_offsets = [offset]

    def __repr__(self):
        return f'UnifiedMarker(labels={self.labels}, body={self.body}, offset={self.offset})'

    def should_merge(self, other: 'UnifiedAnatomicalMarker'):
        return self.body == other.body and np.linalg.norm(self.offset - other.offset) < 0.07

    def merge(self, other: 'UnifiedAnatomicalMarker'):
        assert self.should_merge(other)
        self.labels.extend(other.labels)
        self.merged_offsets.append(other.offset)


class CreateMarkerSuperset(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('create-marker-superset', help='Create a superset of anatomical markers')
        self.register_standard_options(subparser)
        subparser.add_argument('--original-osim-path', type=str, default='../data/unscaled_generic_no_arms.osim',
                               help='The OpenSim generic model which we will output a marker set for.')
        subparser.add_argument('--output-osim-path', type=str, default='../data/markerset.osim',
                               help='The place we will write the OpenSim model with the anatomical markerset.')
        subparser.add_argument('--output-class-map-path', type=str, default='../data/class_map.tsv',
                               help='The output location for the map from file and marker name to output class label.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'create-marker-superset':
            return False
        dataset_home: str = args.dataset_home
        original_osim_path: str = os.path.abspath(args.original_osim_path)
        output_osim_path: str = os.path.abspath(args.output_osim_path)
        output_class_map_path: str = args.output_class_map_path

        data_folders: List[str] = ['dev', 'train']

        subject_paths = []
        for data_folder in data_folders:
            data_path = os.path.join(dataset_home, data_folder)
            if os.path.isdir(data_path):
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith(".b3d") and "vander" not in file.lower():
                            subject_paths.append(os.path.join(root, file))
            else:
                assert data_path.endswith(".b3d")
                subject_paths.append(data_path)

        skeleton_marker_name_to_index: Dict[TrainingMarkerLabel, int] = {}
        num_body_nodes = 0

        # Build the unified anatomical marker set
        unified_anatomical_markers: List[UnifiedAnatomicalMarker] = []

        # Walk the folder path, and check for any with the ".b3d" extension (indicating that they are
        # AddBiomechanics binary data files)
        for i, subject_path in enumerate(subject_paths):
            # Add the skeleton to the list of skeletons
            try:
                subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            except:
                continue
            print('Loading skeleton ' + str(i + 1) + '/' + str(
                len(subject_paths)) + f' for subject {subject_path}')
            osim = subject.readOpenSimFile(0, ignoreGeometry=True)
            skeleton = osim.skeleton
            if skeleton is None:
                continue
            num_body_nodes = skeleton.getNumBodyNodes()
            body_scales: Dict[str, np.ndarray] = osim.bodyScales

            # The anatomical markers get put into a great big list, so we can deduplicate them in the next step,
            # before we assign them to an index in the classification problem.
            for raw_label in osim.anatomicalMarkers:
                marker_label = TrainingMarkerLabel(raw_label, i)
                body_name = osim.markersMap[raw_label][0].getName()
                body_scale = body_scales[body_name]
                marker_offset = osim.markersMap[raw_label][1]
                marker_offset = np.divide(marker_offset, body_scale)
                unified_anatomical_markers.append(UnifiedAnatomicalMarker(marker_label, body_name, marker_offset))

            # The tracking markers get labeled as just the body segment they are attached to
            for raw_label in osim.trackingMarkers:
                marker_label = TrainingMarkerLabel(raw_label, i)
                marker_body_index = osim.markersMap[raw_label][0].getIndexInSkeleton()
                skeleton_marker_name_to_index[marker_label] = marker_body_index

        # Deduplicate the anatomical markers
        anatomical_markers_per_body: Dict[str, List[UnifiedAnatomicalMarker]] = {}
        for i in range(len(unified_anatomical_markers)):
            anatomical_marker = unified_anatomical_markers[i]
            if anatomical_marker.body not in anatomical_markers_per_body:
                anatomical_markers_per_body[anatomical_marker.body] = []
            anatomical_markers_per_body[anatomical_marker.body].append(anatomical_marker)

        for body in anatomical_markers_per_body:
            print(f'Processing body {body}')
            marker_list: List[UnifiedAnatomicalMarker] = anatomical_markers_per_body[body]
            for i in range(len(marker_list)):
                if i > len(marker_list):
                    break
                to_merge: List[int] = []
                for j in range(i + 1, len(marker_list)):
                    # If two markers should merge, merge them, and break out of the loop
                    if marker_list[i].should_merge(marker_list[j]):
                        marker_list[i].merge(marker_list[j])
                        to_merge.append(j)
                for j in reversed(to_merge):
                    marker_list.pop(j)

            # Make sure that no two markers in this body should merge
            for i in range(len(marker_list)):
                marker_list[i].offset = np.mean(marker_list[i].merged_offsets, axis=0)

            # Only keep anatomical markers that are present in at least 4 skeletons
            marker_list = [marker for marker in marker_list if len(marker.merged_offsets) > 3]

            anatomical_markers_per_body[body] = marker_list

        unified_anatomical_markers = []
        for body in anatomical_markers_per_body:
            unified_anatomical_markers.extend(anatomical_markers_per_body[body])
        print(f'Ended up with unified anatomical markers: {len(unified_anatomical_markers)}')

        # Assign a unique index to each marker
        for i, unified_marker in enumerate(unified_anatomical_markers):
            unified_marker.classification_index = num_body_nodes + i
            for label in unified_marker.labels:
                skeleton_marker_name_to_index[label] = unified_marker.classification_index

        # Write the marker superset to disk
        with open(output_class_map_path, 'w') as f:
            f.write('skeleton_path\tmarker_name\tmarker_index\n')
            for key in skeleton_marker_name_to_index:
                f.write(f'{subject_paths[key.skeleton_index]}\t{key.name}\t{skeleton_marker_name_to_index[key]}\n')

        # Write the markerset OpenSim file out
        anatomical_markers: Dict[str, Tuple[str, np.ndarray]] = {}
        is_anatomical_marker = {}
        for i, unified_marker in enumerate(unified_anatomical_markers):
            anatomical_markers[str(unified_marker.classification_index)] = (unified_marker.body, unified_marker.offset)
            is_anatomical_marker[str(unified_marker.classification_index)] = True
        nimble.biomechanics.OpenSimParser.replaceOsimMarkers(original_osim_path, anatomical_markers, is_anatomical_marker, output_osim_path)