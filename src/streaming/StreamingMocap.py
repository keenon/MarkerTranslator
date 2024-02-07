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
    trace_ids: np.ndarray
    while True:
        # Check for new data from the inference process
        if not processing_queue.empty():
            x, trace_ids = processing_queue.get()
            if x.size == 0:
                result_queue.put((np.zeros((0, num_classes)), trace_ids))
                continue
            x = torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)
            mask = torch.ones((1, x.shape[1]), device=device)
            logits = model(x, mask).squeeze(axis=0).to('cpu').detach().numpy()
            result_queue.put((logits.transpose(), trace_ids.transpose()))
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
    stride_ms: int
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
    lab_streaming: nimble.biomechanics.StreamingMocapLab
    markers: List[Tuple[nimble.dynamics.BodyNode, np.ndarray]]

    lines_in_gui: List[str]

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
        self.osim_file.skeleton.setPositionLowerLimits(np.full(self.osim_file.skeleton.getNumDofs(), -1000, dtype=np.float32))
        self.osim_file.skeleton.setPositionUpperLimits(np.full(self.osim_file.skeleton.getNumDofs(), 1000, dtype=np.float32))
        self.weights_path = weights_path
        max_marker_index = max([int(key) for key in self.osim_file.markersMap.keys()])
        self.num_bodies = self.osim_file.skeleton.getNumBodyNodes()
        self.unknown_class_index = max_marker_index + 1
        self.num_classes = max_marker_index + 2
        self.window = int(window)
        self.stride = float(stride)
        self.stride_ms = int(stride * 1000)
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
        self.markers = []
        for i in range(len(self.osim_file.markersMap)):
            key = str(i + self.osim_file.skeleton.getNumBodyNodes())
            assert key in self.osim_file.markersMap
            self.markers.append(self.osim_file.markersMap[key])
        # buffer_size = self.window * int(self.stride / 0.01) * 100
        self.lab_streaming = nimble.biomechanics.StreamingMocapLab(self.osim_file.skeleton, self.markers)
        self.lab_streaming.getMarkerTraces().setMaxJoinDistance(0.15)
        self.lines_in_gui = []

    def set_anthropometrics(self, xml_path: str, data_path: str):
        anthropometrics: nimble.biomechanics.Anthropometrics = nimble.biomechanics.Anthropometrics.loadFromFile(
            xml_path)
        cols = anthropometrics.getMetricNames()
        gauss: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
            data_path,
            cols,
            0.001)  # mm -> m
        # observed_values = {
        #     'stature': self.heightM,
        #     'weightkg': self.massKg * 0.01,
        # }
        # gauss = gauss.condition(observed_values)
        anthropometrics.setDistribution(gauss)
        self.lab_streaming.setAnthropometricPrior(anthropometrics)

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
        # self.gui.nativeAPI().createBox('Center', np.ones(3)*0.5, np.zeros(3), np.zeros(3))
        self.lab_streaming.startGUIThread(self.gui.nativeAPI())

    def start_ik_thread(self):
        self.lab_streaming.startSolverThread()

    def connect_to_cortex(self, cortex_host: str, port: int):
        self.lab_streaming.listenToCortex(cortex_host, port)

    def observe_markers(self, markers: List[np.ndarray], now: float):
        self.lab_streaming.manuallyObserveMarkers(markers, int(now * 1000))
        self.lab_streaming.getMarkerTraces().renderTracesToGUI(self.gui.nativeAPI())

    def run_model(self):
        """
        Run the model on the current traces, if possible, and add the classification logits to the traces estimates
        """
        if not self.inference_process:
            return
        if self.processing_queue.full():
            return

        if not self.result_queue.empty():
            logits, trace_ids = self.result_queue.get()
            self.lab_streaming.observeTraceLogits(logits, trace_ids)

        input_points, trace_ids = self.lab_streaming.getTraceFeatures(
            numWindows=self.window,
            windowDuration=int(self.stride * 1000))
        self.processing_queue.put((input_points.transpose(), trace_ids.transpose()))

    def get_traces(self) -> List[Trace]:
        return self.traces

    def reset(self):
        self.lab_streaming.reset(self.gui.nativeAPI())
