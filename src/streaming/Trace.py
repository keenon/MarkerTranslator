import numpy as np
from typing import List, Dict, Tuple, Optional
from models.TransformerSequenceClassifier import TransformerSequenceClassifier
import torch
import os
import nimblephysics as nimble

class Trace:
    uuid: str
    points: List[np.ndarray]
    times: List[float]
    logits: np.ndarray
    num_bodies: int

    def __init__(self, first_point: np.ndarray, trace_time: float, num_classes: int, num_bodies: int = 0):
        self.points = [first_point]
        self.times = [trace_time]
        self.logits = np.zeros((num_classes,), dtype=np.float32)
        self.num_bodies = num_bodies
        self.uuid = os.urandom(16).hex()

    def add_point(self, point: np.ndarray, time: float):
        self.points.append(point)
        self.times.append(time)

    def project_to(self, time: float):
        if len(self.points) > 1:
            last_vel = (self.points[-1] - self.points[-2]) / (self.times[-1] - self.times[-2])
            projected_now = self.points[-1] + last_vel * (time - self.times[-1])
            return projected_now
        elif len(self.points) == 1:
            return self.points[-1]
        else:
            return np.zeros((3,), dtype=np.float32)

    def dist_to(self, other: np.ndarray):
        return np.linalg.norm(self.points[-1] - other)

    def time_since_last_point(self, now: float):
        return now - self.times[-1]

    def last_time(self):
        return self.times[-1]

    def start_time(self):
        return self.times[0]

    def get_duration(self):
        return self.times[-1] - self.times[0]

    def get_points_at_intervals(self, end: float, interval: float, windows: int) -> List[np.ndarray]:
        start = end - interval * (windows - 1)
        evenly_spaced_times = np.linspace(start, end, windows)

        # Find the indices of the points that are closest to the evenly spaced times, if any
        threshold = 0.01
        points: List[np.ndarray] = []

        points_cursor = 0
        for i in range(len(evenly_spaced_times)):
            target_time = evenly_spaced_times[i]
            while (points_cursor < len(self.times)
                   and abs(self.times[points_cursor] - target_time) > threshold
                   and self.times[points_cursor] < target_time):
                points_cursor += 1
            if points_cursor >= len(self.times):
                break
            if abs(self.times[points_cursor] - target_time) <= threshold:
                augmented_point = np.zeros((4,), dtype=np.float32)
                augmented_point[:3] = self.points[points_cursor]
                augmented_point[3] = i
                points.append(augmented_point)
                points_cursor += 1

        return points

    def drop_from_gui(self, gui: Optional[nimble.gui_server.NimbleGUI]):
        if not gui:
            return
        gui.nativeAPI().deleteObject(self.uuid)

    def render_on_gui(self, gui: Optional[nimble.gui_server.NimbleGUI]):
        if not gui:
            print('NO GUI!!!')
            return
        line_points: List[np.ndarray] = []
        num_points = 20
        if len(self.points) >= num_points:
            line_points = self.points[-num_points:]
        else:
            line_points.extend(self.points)
            line_points.extend([self.points[0] for _ in range(num_points - len(self.points))])
        assert(len(line_points) == num_points)
        max_logit_index = np.argmax(self.logits)
        is_nothing = max_logit_index == len(self.logits) - 1
        is_anatomical = max_logit_index > self.num_bodies
        color = [0.5, 0.5, 0.5, 1.0] if is_nothing else ([0., 0, 1.0, 1.0] if is_anatomical else [1.0, 0., 0., 1.0])
        gui.nativeAPI().createLine(self.uuid, line_points, color)

