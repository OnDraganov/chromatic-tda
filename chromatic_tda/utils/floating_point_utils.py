import numpy as np
from chromatic_tda.utils.singleton import singleton


@singleton
class FloatingPointUtils:

    def is_trivial_bar(self, bar):
        return np.isclose(bar[0], bar[1])


class FloatComparisonWrapper:
    def __init__(self, num):
        self.value = num

    def __lt__(self, other):
        return self.__ne__(other) and self.value < other.value

    def __gt__(self, other):
        return self.__ne__(other) and self.value > other.value

    def __eq__(self, other):
        return np.isclose(self.value, other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.value <= other.value

    def __ge__(self, other):
        return self.__eq__(other) or self.value >= other.value

