import numpy as np
from chromatic_tda.algorithms.miniball.miniball import get_bounding_ball


class MiniballAlgorithm:
    @staticmethod
    def find_smallest_enclosing_ball(points: np.ndarray):
        """:return: center and radius of the smallest enclosing ball"""
        center, rad2 = get_bounding_ball(points)
        return center, np.sqrt(rad2)
