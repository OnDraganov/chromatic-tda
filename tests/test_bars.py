from pathlib import Path
import unittest
import json
import numpy as np

from chromatic_tda import ChromaticAlphaComplex
from chromatic_tda.utils.persistence_utils import PersistenceUtils


# class TestClass(unittest.TestCase):
#
#     def test_something(self):
#         TestBars().test_all()


class TestBars(unittest.TestCase):

    data_folder = Path(__file__).parent / 'test_data'

    def test_all(self):
        print('=== Testing bars computation for all pre-computed tests ===')
        # results = []
        for data_name in ('line_sep', 'one_circle_20_40', 'two_circles', 'two_circles_cc', 'random', 'random2',
                          'one_circle_3col_back_split', 'one_circle_3col_bi-circle_tri-filled',
                          'one_circle_3col_circ_split'):
            test = self.single_test(data_name)
            assert test

    def single_test(self, data_name, return_detailed=False):
        return self.single_test_data(self.load_data(data_name), return_detailed=return_detailed)

    def load_data(self, data_name):
        with open(self.data_folder / f'{data_name}.json', 'r') as file:
            data = json.load(file)

        return data

    def single_test_data(self, data, return_detailed=False):
        results = []
        alpha_complex = ChromaticAlphaComplex(data['points'], data['labels'])
        for instance in data['persistence']:
            results_instance = {}
            simplicial_complex = alpha_complex.get_simplicial_complex(
                complex = instance['map']['complex'],
                sub_complex = instance['map']['sub_complex'],
                relative = instance['map']['relative']
            )
            simplicial_complex.compute_persistence()
            for group in ('kernel', 'sub_complex', 'image', 'complex', 'cokernel', 'relative'):
                bars = simplicial_complex.bars(group)
                bars_test = [(dim, bar) for dim, bar in instance['bars'][group]
                             if not PersistenceUtils().is_trivial_bar(bar)]
                results_instance[group] = all(
                    dim == dim_test and np.isclose(birth, birth_test) and np.isclose(death, death_test)
                    for (dim, (birth, death)), (dim_test, (birth_test, death_test)) in
                    zip(sorted(bars), sorted(bars_test))
                )
            results.append(results_instance)

        if return_detailed:
            return results
        else:
            return all(all(group_result for group_result in instance_results.values()) for instance_results in results)


def main():
    TestBars().test_all()


if __name__ == "__main__":
    main()
