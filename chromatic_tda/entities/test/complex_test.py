import unittest

from chromatic_tda.entities.simplicial_complex import SimplicialComplex

class ComplexTest(unittest.TestCase):
    def test_complex(self, verbose=False) -> None:
        print("===== Testing Simplicial Complex has been started =====")

        complex = SimplicialComplex({(1,2,3),(2,3,4),(3,4,5),(4,6),(5,6),(3,7)})
        if verbose:
            complex.write()

        print(55*"=" + "\n")
