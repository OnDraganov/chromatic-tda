from chromatic_tda.algorithms.radius_function import RadiusFunctionConstructor
from tests.test_bars import TestBars
from chromatic_tda import ChromaticAlphaComplex, plot_six_pack
import numpy as np
from matplotlib import pyplot as plt


def run_test():
    pass
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


def main():
    run_test_plot()


if __name__ == "__main__":
    main()
