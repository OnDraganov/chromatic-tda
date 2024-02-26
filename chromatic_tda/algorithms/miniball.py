import numpy as np
import miniball


class MiniballAlgorithm:
    @staticmethod
    def find_smallest_enclosing_ball(points: np.ndarray):
        """:return: center and radius of the smallest enclosing ball"""
        center, rad2 = miniball.get_bounding_ball(points)
        return center, np.sqrt(rad2)
