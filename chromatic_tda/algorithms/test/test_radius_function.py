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

    def test_2d_rnd_1_2_1(self):
        points = np.array([[0.64200461, 0.04874111],
                           [0.2361311, 0.26006657],
                           [0.90437823, 0.02386989],
                           [0.74969127, 0.7263785]])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(points[:1], points[1:3], points[3:4])
        assert np.isclose([0.63794903, 0.33348898], c).all()
        assert np.isclose(0.28477674882678244, r[0])
        assert np.isclose(0.4084709331753826, r[1])
        assert np.isclose(0.4084709331753826, r[2])

    def test_2d_rnd_3_1_0(self):
        points = np.array([[0.84230833, 0.02261223],
                           [0.99884336, 0.16265999],
                           [0.08682862, 0.20002957],
                           [0.65710032, 0.38482416]])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(points[:3], points[3:4], np.zeros((0, 2)))
        assert np.isclose([0.55590267, 0.50024081], c).all()
        assert np.isclose(0.5569176466310057, r[0])
        assert np.isclose(0.15349907828659298, r[1])
        assert np.isclose(0, r[2])

    def test_2d_rnd_3_1(self):
        points = np.array([[0.84230833, 0.02261223],
                           [0.99884336, 0.16265999],
                           [0.08682862, 0.20002957],
                           [1, 2]])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(points[:3], points[3:4])
        assert np.isclose([0.55590267, 0.50024081], c).all()
        assert np.isclose(0.5569176466310057, r[0])
        assert np.isclose(1.5641291689701118, r[1])

    def test_2d_rnd_1_1_1(self):
        points = np.array([[0.31329716, 0.83100724],
                           [0.01325454, 0.80997717],
                           [0.68344234, 0.39667674]])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(points[:1], points[1:2], points[2:3])
        assert np.isclose([0.34834844, 0.60332695], c).all()
        assert np.isclose(0.23036256029675714, r[0])
        assert np.isclose(0.3936905309541866, r[1])
        assert np.isclose(0.39369052974995866, r[2])

    def test_4d_1_1(self):
        pt1, pt2 = np.array([1, 2, 3, 4]), np.array([5, 4, 5, 4])
        c, r = RadiusFunctionConstructor.find_smallest_circumstack(np.array([pt1]), np.array([pt2]))

        assert np.isclose((pt1 + pt2) / 2, c).all()
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[0])
        assert np.isclose(np.linalg.norm(pt1 - pt2) / 2, r[1])

    # def test_DEBUG(self):  # DEBUGGING
    #     TimingUtils().log_times = True
    #     n = 50
    #     points = np.random.random(size=(n, 4))
    #     labels = [int(3 * np.random.random()) for _ in range(n)]
    #     factory = CoreChromaticAlphaComplexFactory(points, labels)
    #
    #     # print('OLD')
    #     # TimingUtils().flush()
    #     # time_start = time.perf_counter()
    #     # alpha = factory.create_instance(lift_perturbation=1e-9, point_perturbation=None,
    #     #                                 use_morse_optimization=True,
    #     #                                 circumstack_method='miniball',
    #     #                                 old_new_switch='old', )
    #     # print(f'Total time {time.perf_counter() - time_start:.2f} s')
    #     # wf_old = alpha.simplicial_complex.simplex_weights
    #     # TimingUtils().print()
    #
    #     # print()
    #     # print()
    #     print('NEW :: MINIBALL')
    #     print('===============')
    #     TimingUtils().flush()
    #     time_start = time.perf_counter()
    #     alpha = factory.create_instance(lift_perturbation=1e-9, point_perturbation=None,
    #                                     use_morse_optimization=True,
    #                                     circumstack_method='miniball',
    #                                     old_new_switch='new')
    #     print(f'Total time {time.perf_counter() - time_start:.2f} s')
    #     wf_mb = alpha.simplicial_complex.simplex_weights
    #     # print(f'Same keys: {wf_old.keys() == wf_mb.keys()}')
    #     # print(f'Same vals: {all(np.isclose(wf_old[splx], wf_mb[splx]) for splx in wf_old.keys())}')
    #     # print('-----------------------------------------')
    #     TimingUtils().print()
    #
    #     print()
    #     print()
    #     print('NEW :: WEIGHTED CIRCUMSPHERES')
    #     print('===============')
    #     TimingUtils().flush()
    #     time_start = time.perf_counter()
    #     alpha = factory.create_instance(lift_perturbation=1e-9, point_perturbation=None,
    #                                     use_morse_optimization=True,
    #                                     circumstack_method='weighted_circumspheres',
    #                                     old_new_switch='new')
    #     print(f'Total time {time.perf_counter() - time_start:.2f} s')
    #     wf_wc = alpha.simplicial_complex.simplex_weights
    #     # print(f'Same keys: {wf_old.keys() == wf_wc.keys()}')
    #     # print(f'Same vals: {all(np.isclose(wf_old[splx], wf_wc[splx]) for splx in wf_old.keys())}')
    #     # print('-----------------------------------------')
    #     TimingUtils().print()
    #
    #     # print()
    #     # print('-----------------------------------------')
    #     # same_keys = wf_mb.keys() == wf_wc.keys()
    #     # print(f'Same keys: {same_keys}')
    #     # same_vals = all(np.isclose(wf_mb[splx], wf_wc[splx]) for splx in wf_mb.keys())
    #     # print(f'Same vals: {same_vals}')
    #     # print('-----------------------------------------')
    #     #
    #     # print()
    #     # print("simplices")
    #     # print({dim: len(simplices) for dim, simplices in alpha.simplicial_complex.dim_simplex_dict.items()})
    #     # print()
    #     #
    #     # print('agreed')
    #     # agreed = {}
    #     # disagreed = []
    #     # for simplex in wf_mb:
    #     #     dim = len(simplex) - 1
    #     #     if np.isclose(wf_mb[simplex], wf_wc[simplex]):
    #     #         if dim not in agreed:
    #     #             agreed[dim] = 0
    #     #         agreed[dim] += 1
    #     #     else:
    #     #         disagreed.append(simplex)
    #     #
    #     # print(agreed)
    #     # print()
    #     #
    #     # if len(disagreed) > 0:
    #     #     for i in [0, -1]:
    #     #         simplex = disagreed[i]
    #     #         print()
    #     #         print(f'{simplex}  -->  {tuple(alpha.internal_labeling[v] for v in simplex)}')
    #     #         print(wf_mb[simplex])
    #     #         print(wf_wc[simplex])
    #     #         print(alpha.simplicial_complex.co_boundary[simplex])
    #     #
    #     #     print()
    #     #     print()
    #     #     print('==POINTS==')
    #     #     print(alpha.points)
    #     #     print()
    #     #     print()
    #     #     print('==LABELS==')
    #     #     print(alpha.internal_labeling)
    #     #
    #     # assert same_vals and same_keys
