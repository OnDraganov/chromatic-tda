import unittest

import numpy as np

from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor


class RadiusFunctionTest(unittest.TestCase):
    def test_2d_2_1_1(self):
        center, radii = RadiusFunctionConstructor.find_smallest_circumstack(np.array([[1, 1], [-1, -1]]),
                                                                            np.array([[.2, -.1]]),
                                                                            np.array([[6, -4]]))
        assert np.isclose([2.5, -2.5], center).all()
        assert np.isclose(3.807886552931954, radii[0])
        assert np.isclose(3.3241540277189316, radii[1])
        assert np.isclose(radii[0], radii[2])

    def test_2d_2_2(self):
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(
            np.array([[4, 6], [6, 4]]),
            np.array([[-5, 2], [-5, -2]]))
        assert np.isclose([0, 0], c).all()
        assert np.isclose(np.linalg.norm([4, 6]), r[0])
        assert np.isclose(np.linalg.norm([-5, 2]), r[1])

    def test_2d_2_2_1(self):
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(
            np.array([[4, 6], [6, 4]]),
            np.array([[-5, 2], [-5, -2]]),
            np.array([[20, 25]]))
        assert np.isclose([0, 0], c).all()
        assert np.isclose(np.linalg.norm([4, 6]), r[0])
        assert np.isclose(np.linalg.norm([-5, 2]), r[1])
        assert np.isclose(np.linalg.norm([20, 25]), r[2])

    def test_4d_1_1(self):
        pt1, pt2 = np.array([1, 2, 3, 4]), np.array([5, 4, 5, 4])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(np.array([pt1]), np.array([pt2]))

        assert np.isclose((pt1 + pt2) / 2, c).all()
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[0])
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[1])

