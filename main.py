from tests.test_bars import TestBars
from chromatic_tda import ChromaticAlphaComplex, SimplicialComplex
import numpy as np


def main():
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


if __name__ == "__main__":
    main()
