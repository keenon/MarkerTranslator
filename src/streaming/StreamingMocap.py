import numpy as np
from typing import List, Dict, Tuple, Optional
from models.TransformerSequenceClassifier import TransformerSequenceClassifier
import torch
import os
import nimblephysics as nimble
from streaming.Trace import Trace
import time
import multiprocessing


def slow_inference_process(
        weights_path: str,
        # Queues
        processing_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        # Model output sizes
        num_classes: int,
        # Model parameters
        d_model=256,
        nhead=8,
        num_transformer_layers=6,
        dim_feedforward=256,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32):
    model: TransformerSequenceClassifier
    model = TransformerSequenceClassifier(num_classes=num_classes,
                                          d_model=d_model,
                                          nhead=nhead,
                                          num_transformer_layers=num_transformer_layers,
                                          dim_feedforward=dim_feedforward,
                                          device=device,
                                          dtype=dtype)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))['model_state_dict'])
    model.eval()

    x: np.ndarray
    points_to_trace_uuids: List[str]
    while True:
        # Check for new data from the inference process
        if not processing_queue.empty():
            x, points_to_trace_uuids = processing_queue.get()
            x = torch.tensor(np.stack(x), device=device).unsqueeze(0)
            mask = torch.ones((1, x.shape[1]), device=device)
            logits = model(x, mask).to('cpu').detach().numpy()
            result_queue.put((logits, points_to_trace_uuids))
        time.sleep(0.1)


class StreamingMocap:
    traces: List[Trace]
    osim_file: nimble.biomechanics.OpenSimFile
    gui: Optional[nimble.gui_server.NimbleGUI]
    weights_path: str
    num_bodies: int
    num_classes: int
    unknown_class_index: int
    window: int
    stride: float
    cut_off_time: float
    device: str
    processing_queue: multiprocessing.Queue
    inference_process: Optional[multiprocessing.Process]
    result_queue: multiprocessing.Queue
    d_model: int
    nhead: int
    num_transformer_layers: int
    dim_feedforward: int
    device: str
    dtype: torch.dtype

    def __init__(self,
                 # Model paths
                 unscaled_generic_model_path: str,
                 geometry_path: str,
                 weights_path: str,
                 # Model input details
                 window: int = 10,
                 stride: float = 0.15,
                 # Model parameters
                 d_model=256,
                 nhead=8,
                 num_transformer_layers=6,
                 dim_feedforward=256,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.traces = []
        self.osim_file = nimble.biomechanics.OpenSimParser.parseOsim(os.path.abspath(unscaled_generic_model_path), geometryFolder=os.path.abspath(geometry_path)+'/' if len(geometry_path) > 0 else '')
        self.weights_path = weights_path
        max_marker_index = max([int(key) for key in self.osim_file.markersMap.keys()])
        self.num_bodies = self.osim_file.skeleton.getNumBodyNodes()
        self.unknown_class_index = max_marker_index + 1
        self.num_classes = max_marker_index + 2
        self.window = int(window)
        self.stride = float(stride)
        self.gui = None
        # Drop traces that haven't been updated in this many seconds
        self.cut_off_time = 0.5
        self.processing_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        self.inference_process = None
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.device = device
        self.dtype = dtype

    def start_inference_process(self):
        self.inference_process = multiprocessing.Process(target=slow_inference_process,
                                                         daemon=False,
                                                         args=(self.weights_path,
                                                               self.processing_queue,
                                                               self.result_queue,
                                                               self.num_classes,
                                                               self.d_model,
                                                               self.nhead,
                                                               self.num_transformer_layers,
                                                               self.dim_feedforward,
                                                               self.device,
                                                               self.dtype))
        self.inference_process.start()

    def start_gui(self):
        self.gui = nimble.gui_server.NimbleGUI()
        self.gui.serve(8080)

    def run_ik_update(self):
        # 1. Work out which traces are currently classified as anatomical markers
        markers: Dict[str, np.ndarray] = {}
        # print('Running IK update')
        for i, trace in enumerate(self.traces):
            if not trace.logits.nonzero():
                # print(i, 'Zero logits trace')
                continue
            max_logit_index = np.argmax(trace.logits)
            if max_logit_index < self.num_bodies:
                # print(i, 'Body trace ', max_logit_index)
                continue
            if max_logit_index == self.unknown_class_index:
                # print(i, 'Unknown trace')
                continue
            # print(i, 'Adding marker', max_logit_index, 'to IK')
            markers[str(max_logit_index)] = trace.points[-1]
        if len(markers) < 3:
            return
        # 2. Run IK
        marker_list: List[Tuple[nimble.dynamics.BodyNode, np.ndarray]] = []
        marker_positions: np.ndarray = np.zeros(len(markers) * 3, dtype=np.float32)
        marker_weights: np.ndarray = np.ones(len(markers), dtype=np.float32)
        marker_cursor = 0
        for key in markers.keys():
            if key not in self.osim_file.markersMap:
                print('Marker', key, 'not found in osim file, which has ', self.osim_file.markersMap.keys(), 'markers')
                continue
            marker_list.append(self.osim_file.markersMap[key])
            marker_positions[marker_cursor:marker_cursor + 3] = markers[key]
            marker_cursor += 3
        print('Running IK with markers', markers.keys())
        self.osim_file.skeleton.fitMarkersToWorldPositions(marker_list, marker_positions, marker_weights, scaleBodies=True, maxStepCount=10, lineSearch=True)
        # 3. Update the gui
        if self.gui:
            print('Rendering skeleton')
            self.gui.nativeAPI().renderSkeleton(self.osim_file.skeleton)

    def observe_markers(self, markers: List[np.ndarray], now: float):
        # 1. Trim the old traces
        drop_traces = [trace for trace in self.traces if trace.time_since_last_point(now) >= self.cut_off_time]
        for trace in drop_traces:
            trace.drop_from_gui(self.gui)
        self.traces = [trace for trace in self.traces if trace.time_since_last_point(now) < self.cut_off_time]
        # 2. Assign markers to traces
        # 2.0. Pre-compute the distances between all markers and traces
        markers_assigned: List[bool] = [False for _ in markers]
        dists: np.ndarray = np.zeros((len(self.traces), len(markers)), dtype=np.float32)
        for i, trace in enumerate(self.traces):
            projected_marker = trace.project_to(now)
            for j, marker in enumerate(markers):
                dists[i, j] = np.linalg.norm(projected_marker - marker)
        # 2.1. Greedily assign the closest pair together, until we have no pairs left
        for k in range(min(len(markers), len(self.traces))):
            # Find the closest pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            dist = dists[i, j]
            if dist > 0.1:
                break
            # Add the marker to the trace
            markers_assigned[j] = True
            self.traces[i].add_point(markers[j], now)
            dists[i, :] = np.inf
            dists[:, j] = np.inf
        # 3. Add any remaining markers as new traces
        for j in range(len(markers)):
            if not markers_assigned[j]:
                self.traces.append(Trace(markers[j], now, self.num_classes, self.num_bodies))
        # 4. Render the traces on the gui
        for trace in self.traces:
            trace.render_on_gui(self.gui)

    def run_model(self):
        """
        Run the model on the current traces, if possible, and add the classification logits to the traces estimates
        """
        if not self.inference_process:
            return
        if self.processing_queue.full():
            return

        start_compute_time = time.time()

        if not self.result_queue.empty():
            logits, points_to_trace_uuids = self.result_queue.get()

            # Add the last logits to the traces
            trace_dict = {trace.uuid: trace for trace in self.traces}
            for i in range(len(points_to_trace_uuids)):
                trace_uuid = points_to_trace_uuids[i]
                if trace_uuid not in trace_dict:
                    continue
                trace = trace_dict[trace_uuid]
                # trace.logits = logits[0, i, :] * 0.5 + trace.logits * 0.5
                trace.logits = logits[0, i, :]

        # Get the traces that are long enough to run the model on
        if len(self.traces) == 0:
            return
        expected_duration = self.window * self.stride
        now: float = max([trace.last_time() for trace in self.traces])
        start: float = min([trace.start_time() for trace in self.traces])
        if now - start < expected_duration:
            return

        input_points: List[np.ndarray] = []
        points_to_trace_uuids: List[str] = []

        for i in range(len(self.traces)):
            trace = self.traces[i]
            points = trace.get_points_at_intervals(now, self.stride, self.window)
            input_points.extend(points)
            points_to_trace_uuids.extend([trace.uuid for _ in range(len(points))])

        x = np.stack(input_points)
        # Center the first 3 rows
        x[:, :3] -= x[:, :3].mean(axis=0)
        self.processing_queue.put((x, points_to_trace_uuids))
        print('Time to compute inputs:', time.time() - start_compute_time)

    def get_traces(self) -> List[Trace]:
        return self.traces

    def reset(self):
        for trace in self.traces:
            trace.drop_from_gui(self.gui)
        self.traces = []
