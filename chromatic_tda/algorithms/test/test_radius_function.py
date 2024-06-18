import unittest

import numpy as np
from matplotlib import pyplot as plt  # DEBUGGING
import time

from chromatic_tda import plot_six_pack, SimplicialComplex  # DEBUGGING
from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor
from chromatic_tda.core.chromatic_alpha_complex_factory import CoreChromaticAlphaComplexFactory  # DEBUGGING
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex  # DEBUGGING
from chromatic_tda.utils.timing import TimingUtils  # DEBUGGING


class RadiusFunctionTest(unittest.TestCase):
    def test_2d_2_1_1(self):
        center, radii2 = RadiusFunctionConstructor.find_smallest_circumstack(np.array([[1, 1], [-1, -1]]),
                                                                             np.array([[.2, -.1]]),
                                                                             np.array([[6, -4]]))
        radii = np.sqrt(radii2)
        assert np.isclose([2.5, -2.5], center).all()
        assert np.isclose(3.807886552931954, radii[0])
        assert np.isclose(3.3241540277189316, radii[1])
        assert np.isclose(radii[0], radii[2])

    def test_2d_2_2(self):
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(
            np.array([[4, 6], [6, 4]]),
            np.array([[-5, 2], [-5, -2]]))
        r = np.sqrt(r2)
        assert np.isclose([0, 0], c).all()
        assert np.isclose(np.linalg.norm([4, 6]), r[0])
        assert np.isclose(np.linalg.norm([-5, 2]), r[1])

    def test_2d_2_2_1(self):
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(
            np.array([[4, 6], [6, 4]]),
            np.array([[-5, 2], [-5, -2]]),
            np.array([[20, 25]]))
        r = np.sqrt(r2)
        assert np.isclose([0, 0], c).all()
        assert np.isclose(np.linalg.norm([4, 6]), r[0])
        assert np.isclose(np.linalg.norm([-5, 2]), r[1])
        assert np.isclose(np.linalg.norm([20, 25]), r[2])

    def test_2d_rnd_1_2_1(self):
        points = np.array([[0.64200461, 0.04874111],
                           [0.2361311, 0.26006657],
                           [0.90437823, 0.02386989],
                           [0.74969127, 0.7263785]])
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(points[:1], points[1:3], points[3:4])
        r = np.sqrt(r2)
        assert np.isclose([0.63794903, 0.33348898], c).all()
        assert np.isclose(0.28477674882678244, r[0])
        assert np.isclose(0.4084709331753826, r[1])
        assert np.isclose(0.4084709331753826, r[2])

    def test_2d_rnd_3_1(self):
        points = np.array([[0.84230833, 0.02261223],
                           [0.99884336, 0.16265999],
                           [0.08682862, 0.20002957],
                           [1, 2]])
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(points[:3], points[3:4])
        r = np.sqrt(r2)
        assert np.isclose([0.55590267, 0.50024081], c).all()
        assert np.isclose(0.5569176466310057, r[0])
        assert np.isclose(1.5641291689701118, r[1])

    def test_2d_rnd_3_1_case2(self):
        points = np.array([[0.84230833, 0.02261223],
                           [0.99884336, 0.16265999],
                           [0.08682862, 0.20002957],
                           [0.65710032, 0.38482416]])
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(points[:3], points[3:4])
        r = np.sqrt(r2)
        assert np.isclose([0.55590267, 0.50024081], c).all()
        assert np.isclose(0.5569176466310057, r[0])
        assert np.isclose(0.15349907828659298, r[1])

    def test_2d_rnd_1_1_1(self):
        points = np.array([[0.31329716, 0.83100724],
                           [0.01325454, 0.80997717],
                           [0.68344234, 0.39667674]])
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(points[:1], points[1:2], points[2:3])
        r = np.sqrt(r2)
        assert np.isclose([0.34834844, 0.60332695], c).all()
        assert np.isclose(0.23036256029675714, r[0])
        assert np.isclose(0.3936905309541866, r[1])
        assert np.isclose(0.39369052974995866, r[2])

    def test_4d_1_1(self):
        pt1, pt2 = np.array([1, 2, 3, 4]), np.array([5, 4, 5, 4])
        c, r2 = RadiusFunctionConstructor.find_smallest_circumstack(np.array([pt1]), np.array([pt2]))
        r = np.sqrt(r2)

        assert np.isclose((pt1 + pt2) / 2, c).all()
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[0])
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[1])