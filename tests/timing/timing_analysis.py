import random

from chromatic_tda.entities.chromatic_alpha_complex import ChromaticAlphaComplex
from chromatic_tda.entities import SimplicialComplex
from chromatic_tda.utils.timing import TimingUtils


class TimingAnalysis:

    def __init__(self, n=20, color_range_splits=(.5, 1), complex=None, sub_complex='0', relative=None):
        # Define points and labels
        self.n = n  # number of points
        self.complex = complex
        self.sub_complex = sub_complex
        self.relative = relative
        self.points = [tuple(random.random() for _ in range(2)) for _ in range(n)]  # generate points in [0,1)^2
        self.point_labels = [0 if val < color_range_splits[0] else (1 if val < color_range_splits[1] else 2)
                             for val in [random.random() for _ in range(len(self.points))]]  # label points

    def run(self) -> None:
        print(f"===== Delaunay Complex Timing test with {self.n} points, {len(set(self.point_labels))} colors =====")
        TimingUtils().start("Total")

        TimingUtils().start("Init Complex")
        delaunay_complex = ChromaticAlphaComplex(self.points, self.point_labels)
        TimingUtils().stop("Init Complex")

        TimingUtils().start("Get Complex")
        simplicial_complex : SimplicialComplex = delaunay_complex.get_simplicial_complex(
            sub_complex=self.sub_complex,
            complex=self.complex,
            relative=self.relative
        )
        TimingUtils().stop("Get Complex")

        TimingUtils().start("Compute Persistence")
        simplicial_complex.compute_persistence()
        TimingUtils().stop("Compute Persistence")

        TimingUtils().stop("Total")
        TimingUtils().print()
        print(55*"=" + "\n")


if __name__ == "__main__":
    TimingAnalysis(1000).run()
