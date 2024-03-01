from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor
from tests.test_bars import TestBars
from chromatic_tda import ChromaticAlphaComplex, plot_six_pack
import numpy as np
from matplotlib import pyplot as plt


def run_test():
    TestBars().test_all(verbose=True, assertions=False)
    # for embedding in TestBars().single_test('two_circles_cc', return_detailed=True):
    #     for group, result in embedding.items():
    #         print(group.ljust(12), result)
    #     print()

    # points = np.random.random((50, 2))
    # labels = list(map(int, 2 * np.random.random(len(points))))
    # cplx = ChromaticAlphaComplex(points, labels).get_simplicial_complex(sub_complex=((0,),))

    # TimingAnalysis(
    #     n=1000,
    #     color_range_splits=(.5, 1),
    #     sub_complex='monochromatic'
    # ).run()
    # TimingUtils().flush()
    #
    # TimingAnalysis(
    #     n=500,
    #     color_range_splits=(.3, .6),
    #     sub_complex='monochromatic'
    # ).run()


def run_test_plot():
    points = np.random.random((50, 2))
    labels = list(map(int, 2 * np.random.random(len(points))))
    cplx = ChromaticAlphaComplex(points, labels).get_simplicial_complex(sub_complex='0')
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    plot_six_pack(cplx, axs = axs)
    plt.show()


def run_miniball_changes_DEBUG():
    point_sets = (np.array([[0.48065001, 0.43088975, 0.52390577, 0.20300073],
                            [0.57889628, 0.36719648, 0.71005921, 0.81086358],
                            [0.0816823, 0.30108543, 0.51292335, 0.48475136]]),
                  np.array([[0.17193163, 0.29165913, 0.39361874, 0.63136425],
                            [0.11843034, 0.42468815, 0.5880888, 0.3374743]]),
                  np.array([[0.18185465, 0.72902313, 0.3922063, 0.42804651]]))
    radii2 = np.array([0.14054122, 0.12550183, 0.14054122])
    for _ in range(200):
        print('> ', end='')
        center_mb, radii2_mb = RadiusFunctionConstructor.find_smallest_circumstack(
            *point_sets, circumstack_method='miniball')
        if not np.allclose(np.sqrt(max(radii2_mb)), np.sqrt(max(radii2))):
            print(np.sqrt(max(radii2_mb)), end='; ')
            for pts, rad2 in zip(point_sets, radii2_mb):
                for pt in pts:
                    print(np.isclose(((pt - center_mb) ** 2).sum(), rad2), end=' ')
                print('|', end=' ')
        print()
    print('all done')


def main():
    run_miniball_changes_DEBUG()


if __name__ == "__main__":
    main()
