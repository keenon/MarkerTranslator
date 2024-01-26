import unittest
import numpy as np
import os
from src.streaming.Trace import Trace


class TestTrace(unittest.TestCase):
    def setUp(self):
        self.first_point = np.array([0, 0])
        self.trace_time = 0.0
        self.num_classes = 10
        self.trace = Trace(self.first_point, self.trace_time, self.num_classes)

    def test_initialization(self):
        # Test initialization logic
        self.assertTrue(np.array_equal(self.trace.points[0], self.first_point))
        self.assertEqual(self.trace.times[0], self.trace_time)
        self.assertEqual(len(self.trace.logits), self.num_classes)
        self.assertTrue(isinstance(self.trace.uuid, str) and len(self.trace.uuid) == 32)  # UUID is a 32-char hex string

    def test_add_point(self):
        # Test adding a point
        new_point = np.array([1, 1, 1])
        new_time = 1.0
        self.trace.add_point(new_point, new_time)

        self.assertTrue(np.array_equal(self.trace.points[-1], new_point))
        self.assertEqual(self.trace.times[-1], new_time)

    def test_dist_to(self):
        # Test distance calculation
        other_point = np.array([3, 4, 0])
        self.trace.add_point(np.array([0, 0, 0]), 1.0)  # Add origin point for simplicity
        distance = self.trace.dist_to(other_point)

        self.assertEqual(distance, 5.0)  # Distance from (0,0) to (3,4) is 5.0

    def test_time_since_last_point(self):
        # Test time since last point added
        self.trace.add_point(np.array([1, 1, 1]), 1.0)
        time_since_last = self.trace.time_since_last_point(2.0)

        self.assertEqual(time_since_last, 1.0)

    def test_get_duration(self):
        # Test get_duration method
        self.trace.add_point(np.array([1, 1, 1]), 1.0)
        self.trace.add_point(np.array([2, 2, 2]), 2.0)

        duration = self.trace.get_duration()
        self.assertEqual(duration, 2.0)

    def test_points_at_intervals(self):
        # Add points at evenly spaced intervals
        interval = 1.0  # 1 second interval
        for i in range(1, 6):  # Adding points at 1, 2, 3, 4, 5 seconds
            point = np.array([i, i, i])  # Creating a new point
            self.trace.add_point(point, i * interval)

        # Request points at the same intervals they were added
        windows = 5  # Number of intervals
        end_time = 5.0  # End time is at 5 seconds

        points = self.trace.get_points_at_intervals(end_time, interval, windows)

        # Check if all the points are returned
        expected_points = [np.array([i, i, i, i-1]) for i in range(1, 6)]

        # Check if the number of returned points is correct
        self.assertEqual(len(points), len(expected_points))

        for point, expected_point in zip(points, expected_points):
            self.assertTrue(np.array_equal(point, expected_point))

    def test_points_at_intervals_missing_some(self):
        # Add points at evenly spaced intervals
        interval = 1.0  # 1 second interval
        expected_points = []
        for i in range(1, 6):  # Adding points at 1, 2, 3, 4, 5 seconds
            if i % 2 == 0:
                continue
            point = np.array([i, i, i])  # Creating a new point
            self.trace.add_point(point, i * interval)
            expected_points.append(np.array([i, i, i, i-1]))

        # Request points at the same intervals they were added
        windows = 5  # Number of intervals
        end_time = 5.0  # End time is at 5 seconds

        points = self.trace.get_points_at_intervals(end_time, interval, windows)

        # Check if the number of returned points is correct
        self.assertEqual(len(points), len(expected_points))

        for point, expected_point in zip(points, expected_points):
            self.assertTrue(np.array_equal(point, expected_point))


if __name__ == '__main__':
    unittest.main()