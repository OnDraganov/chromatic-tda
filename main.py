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
    # run_test_plot()
    points = (
        [0.09459416951153077, 0.11232276848750489],  # 0
        [0.9680112771867585, 0.15811534146081496],   # 1
        [0.9658547896521128, 0.1702045941509115],    # 2
        [0.8312718291955721, 0.6406528236300497],    # 3
        [0.09659691184397712, 0.11313859402162363],  # 4
        [0.9595099529549772, 0.15846022651540637],   # 5
        [0.8425246776313106, 0.635219429110198],     # 6
        [0.9701160920689117, 0.16062702956075603],   # 7
        [0.8384241285861569, 0.6479053787601031],    # 8
        [0.10089735680674439, 0.12333065671267851],  # 9
        [0.831030327917524, 0.6343887895213592],     # 10
        [0.08917225973887621, 0.1294156525294371]    # 11
    )
    labels = (0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)
    cplx = ChromaticAlphaComplex(points, labels).get_simplicial_complex(sub_complex='0')
    cplx.compute_persistence()
    cplx_2_bars = [(b, d) for b, d in cplx.core_complex.birth_death['complex']['pairs']
                   if cplx.weight_function(b) < cplx.weight_function(d)
                   and len(b) == 3]
    print(cplx_2_bars)  # [((0, 9, 11), (0, 4, 9, 11)), ((2, 5, 7), (1, 2, 5, 7))]


if __name__ == "__main__":
    main()
