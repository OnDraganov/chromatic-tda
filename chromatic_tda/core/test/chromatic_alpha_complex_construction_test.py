import unittest
from pathlib import Path
import json
import numpy as np

from chromatic_tda.core.chromatic_alpha_complex_factory import CoreChromaticAlphaComplexFactory
from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex


class ChromaticAlphaComplexConstructionTest(unittest.TestCase):

    data_folder = Path(__file__).parent / 'test_data'

    def test_chromatic_alpha_random_10_10(self):
        result = self.single_test(data_name='chralph_random_10_10')
        assert result

    def test_chromatic_alpha_random_10_10_5(self):
        result = self.single_test(data_name='chralph_random_10_10_5')
        assert result

    def test_chromatic_alpha_random_5_2_1(self):
        result = self.single_test(data_name='chralph_random_5_2_1')
        assert result

    def test_chromatic_alpha_random_15_15(self):
        result = self.single_test(data_name='chralph_random_15_15')
        assert result

    def test_chromatic_alpha_random_15_8_7(self):
        result = self.single_test(data_name='chralph_random_15_8_7')
        assert result

    def test_chromatic_alpha_integer_2_2(self):
        result = self.single_test(data_name='chralph_integer_2_2')
        assert result

    def single_test(self, data_name, data_folder=None):
        data = self.load_data(data_name, data_folder)
        factory = CoreChromaticAlphaComplexFactory(points=data['points'], labels=data['labels'])
        alpha = factory.create_instance(lift_perturbation=1e-9, point_perturbation=None)
        return self.compare_complex(alpha, data['weight_function'])

    def load_data(self, data_name, data_folder):
        if data_folder is None:
            data_folder = self.data_folder
        with open(data_folder / (data_name + '.json'), 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def compare_complex(alpha: CoreChromaticAlphaComplex, reference_weight_function):
        computed_simplices = set(alpha.simplicial_complex.get_simplices())
        reference_simplices = set(tuple(sorted(item['simplex'])) for item in reference_weight_function)
        if computed_simplices != reference_simplices:
            return False
        for item in reference_weight_function:
            computed_weight = alpha.simplicial_complex.get_simplex_weight(tuple(sorted(item['simplex'])))
            reference_weight = item['weight']
            if not np.isclose(computed_weight, reference_weight):
                return False
        return True  # simplices identical to the reference, and all the weights are close to the reference
