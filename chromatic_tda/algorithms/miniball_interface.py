import numpy as np
from chromatic_tda.algorithms.miniball.miniball import get_bounding_ball
from chromatic_tda.utils.timing import TimingUtils


class MiniballAlgorithm:
    @staticmethod
    def find_smallest_enclosing_ball(points: np.ndarray):
        """:return: center and radius of the smallest enclosing ball"""
        TimingUtils().start("Miniball")
        center, rad2 = get_bounding_ball(points, check_circumsphere_solutions=False)
        TimingUtils().stop("Miniball")
        return center, np.sqrt(rad2)
