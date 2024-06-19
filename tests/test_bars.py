from pathlib import Path
import unittest
import json
import numpy as np

from chromatic_tda import ChromaticAlphaComplex
from chromatic_tda.utils.floating_point_utils import FloatingPointUtils


# class TestClass(unittest.TestCase):
#
#     def test_something(self):
#         TestBars().test_all()


class TestBars(unittest.TestCase):

    data_folder = Path(__file__).parent / 'test_data'

    def test_precomputed_line_sep(self, verbose=False, assertions=True):
        self.single_test('line_sep', verbose, assertions)

    def test_precomputed_one_circle_20_40(self, verbose=False, assertions=True):
        self.single_test('one_circle_20_40', verbose, assertions)

    def test_precomputed_two_circles(self, verbose=False, assertions=True):
        self.single_test('two_circles', verbose, assertions)

    def test_precomputed_two_circles_cc(self, verbose=False, assertions=True):
        self.single_test('two_circles_cc', verbose, assertions)

    def test_precomputed_random(self, verbose=False, assertions=True):
        self.single_test('random', verbose, assertions)

    def test_precomputed_random2(self, verbose=False, assertions=True):
        self.single_test('random2', verbose, assertions)

    # def test_precomputed_one_circle_3col_back_split(self, verbose=False, assertions=True):
    #     self.single_test('one_circle_3col_back_split', verbose, assertions)
    #
    # def test_precomputed_one_circle_3col_circ_split(self, verbose=False, assertions=True):
    #     self.single_test('one_circle_3col_circ_split', verbose, assertions)

    def test_precomputed_one_circle_3col_bi_circle_tri_filled(self, verbose=False, assertions=True):
        self.single_test('one_circle_3col_bi-circle_tri-filled', verbose, assertions)

    def test_custom_colors(self, verbose=False, assertions=True):
        # an instance with labels 'blue', 'orange' instead of 0, 1
        data = self.load_data('one_circle_20_40')
        data['labels'] = ['blue' if lab == 0 else 'orange' for lab in data['labels']]
        for instance in data['persistence']:
            instance['map']['sub_complex'] = [
                [{'0': 'blue', '1': 'orange'}[ch]] for ch in instance['map']['sub_complex'].split(',')]
        test = self.single_test_data(data, return_detailed=False)
        if assertions:
            assert test
        if verbose:
            print(f'    {test}  ...  one_circle_20_40 recolored')

    def test_close_stack_radii_artifact_2hom(self):
        points = (
            [0.09459416951153077, 0.11232276848750489],  # 0
            [0.9680112771867585, 0.15811534146081496],   # 1
            [0.9658547896521128, 0.1702045941509115],    # 2
            [0.8312718291955721, 0.6406528236300497],    # 3
            [0.09659691184397712, 0.11313859402162363],  # 4
            [0.9595099529549772, 0.15846022651540637],   # 5
            [0.8425246776313106, 0.635219429110198],     # 6
            [0.9701160920689117, 0.16062702956075603],   # 7
            [0.8384241285861569, 0.6479053787601031],    # 8
            [0.10089735680674439, 0.12333065671267851],  # 9
            [0.831030327917524, 0.6343887895213592],     # 10
            [0.08917225973887621, 0.1294156525294371]    # 11
        )
        labels = (0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)
        cplx = ChromaticAlphaComplex(points, labels).get_simplicial_complex(sub_complex='0')
        cplx.compute_persistence()
        cplx_2_bars = [(b, d) for b, d in cplx.core_complex.birth_death['complex']['pairs']
                       if cplx.weight_function(b) < cplx.weight_function(d)
                       and len(b) == 3]
        assert len(cplx_2_bars) == 0

    def single_test(self, data_name, verbose=False, assertions=True):
        test = self.single_test_data(self.load_data(data_name), return_detailed=False)
        if assertions:
            assert test
        if verbose:
            print(f'    {test}  ...  {data_name}')

    def load_data(self, data_name):
        with open(self.data_folder / f'{data_name}.json', 'r') as file:
            data = json.load(file)

        return data

    @staticmethod
    def single_test_data(data, return_detailed=False):
        results = []
        alpha_complex = ChromaticAlphaComplex(data['points'], data['labels'])
        for instance in data['persistence']:
            results_instance = {}
            simplicial_complex = alpha_complex.get_simplicial_complex(
                full_complex = instance['map']['complex'],
                sub_complex = instance['map']['sub_complex'],
                relative = instance['map']['relative']
            )
            simplicial_complex.compute_persistence()
            for group in ('kernel', 'sub_complex', 'image', 'complex', 'cokernel', 'relative'):
                bars = simplicial_complex.bars(group, return_as='list')
                bars_test = [(dim, bar) for dim, bar in instance['bars'][group]
                             if not FloatingPointUtils.is_trivial_bar(bar)]
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
    pass


if __name__ == "__main__":
    main()
