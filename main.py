from tests.test_bars import TestBars
from chromatic_tda import ChromaticAlphaComplex, SimplicialComplex
import numpy as np


def main():
    TestBars().test_all(verbose=True, assertions=False)

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
