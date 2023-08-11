import numpy as np
from chromatic_tda.utils.singleton import singleton


@singleton
class PersistenceUtils:

    def is_trivial_bar(self, bar):
        return np.isclose(bar[0], bar[1])