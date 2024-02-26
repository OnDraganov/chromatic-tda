import unittest

import numpy as np

from chromatic_tda.algorithms.miniball import MiniballAlgorithm


class MiniballTest(unittest.TestCase):
    def test_grid(self):
        points = np.array([(150, 650), (150, 700), (200, 600), (200, 650),
                           (250, 600), (250, 650), (100, 700), (250, 700)])
        center, rad = MiniballAlgorithm.find_smallest_enclosing_ball(points)
        assert np.isclose([175., 650.], center).all()
        assert np.isclose(np.sqrt(8125), rad)

    def test_4d_edge(self):
        points = np.array([[0.74738687, 0.38341592, 0.12021395, 0.08205199],
                           [1.19875894, 0.59553592, 0.27018718, 0.73280497],
                           [1.08064998, 1.01296962, 0.2023605, 0.79952933],
                           [1.85641092, 0.23731939, 0.27123382, 0.27210152],
                           [0.60467603, 0.62813571, 0.97253083, 0.87040604],
                           [0, 0, 0, 0],
                           [1.41656959, 0.04856956, 0.11223752, 1.20596297],
                           [0.7476833, 0.14920541, 1.41493835, 0.51065473],
                           [0.7390609, 0.76201075, 1.03443779, 0.170946],
                           [2, 2, 2, 2],
                           [0.61666966, 0.25993983, 1.28237657, 1.02298014],
                           [0.88682694, 1.38331382, 0.24826776, 0.09740208]])
        center, rad = MiniballAlgorithm.find_smallest_enclosing_ball(points)
        assert np.isclose([1, 1, 1, 1], center).all()
        assert np.isclose(2, rad)

    def test_4d_many_on_boundary(self):
        points = np.array([[0.17602698, 0.95701494, 0.3912977, 0.83253401],
                           [0.48269993, 0.38847898, 0.7428899, 1.22592943],
                           [1.74890368, 0.45470659, 0.70119193, 0.33101947],
                           [1.77406677, 0.62748893, 0.56815702, 0.19501796],
                           [1.35049625, 0.52389997, 0.66202051, 1.13657506],
                           [0.71972512, 1.03775318, 1.35387453, 0.11573636],
                           [0.20601388, 1.09189058, 0.70476006, 0.80286877],
                           [0.8602786, 0.07029879, 1.00197885, 0.4517794],
                           [1.32716285, 0.21625287, 0.70441182, 1.02192401],
                           [0.33842441, 1.36549566, 0.53931755, 1.29045377],
                           [0.34228532, 0.25093733, 0.27001813, 0.88760673]])
        points = np.concatenate([points,
                                 [[3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 1, 3],
                                  [0, 0, 0, 0], [1, 1, 1, 1], [1 + np.sqrt(2), 1 + np.sqrt(2), 1, 1],
                                  [1, 1, 1 + np.sqrt(2), 1 + np.sqrt(2)], [1, 1 + np.sqrt(2), 1, 1 + np.sqrt(2)],
                                  [1 + np.sqrt(2), 1, 1 + np.sqrt(2), 1], [1, 1 + np.sqrt(2), 1 + np.sqrt(2), 1]]])
        center, rad = MiniballAlgorithm.find_smallest_enclosing_ball(points)
        assert np.isclose([1, 1, 1, 1], center).all()
        assert np.isclose(2, rad)

    def test_2d_simple_stack(self):
        points = np.array([[1, 1], [-1, -1], [.2, -.1], [.1, -.2], [6, -4], [4, -6]])
        center, rad = MiniballAlgorithm.find_smallest_enclosing_ball(points)
        assert np.isclose((points[0] + points[-1]) / 2, center).all()

