from chromatic_tda.core.core_chromatic_alpha_complex import CoreChromaticAlphaComplex
from chromatic_tda.core.chromatic_alpha_complex_factory import CoreChromaticAlphaComplexFactory
from chromatic_tda.entities.simplicial_complex import SimplicialComplex


class ChromaticAlphaComplex:

    def __init__(self, points, labels, lift_perturbation=1e-9, point_perturbation=None, **kwargs) -> None:
        """Create an instance of ChromaticAlphaComplex. The object contains the full chromatic Delaunay complex together
        with its alpha radius function.

        Arguments:
            points ... List of point coordinates
            labels ... List of labels. The length has to be the same as the number of points.
                       The first label goes with the first point, etc... The labels can be any hashable object.

        Keyword arguments:
            point_perturbation ... Perturb the points given on the input by +-point_perturbation/2 in each coordinate.
                                   (default: None)
            lift_perturbation ... Compute the Delaunay complex with perturbed lifting to break the non-general position.
                                  Generally leads to faster computation, as QHull does not need to deal with the
                                  non-generality itself. (default: 1e-9)
        """
        self.core_alpha_complex : CoreChromaticAlphaComplex = CoreChromaticAlphaComplexFactory().create_instance(
            points, labels, lift_perturbation, point_perturbation, **kwargs)

    def get_simplicial_complex(self, sub_complex=None, complex=None, relative=None, allow_unused_labels=False) -> SimplicialComplex:
        """Generate a simplicial complex and sub-complex pair based on the parameters given.
        The parameter complex restricts the complex, the parameter sub_complex defines the sub-complex,
        and the parameter relative erases simplices from the complex to represent a relative simplicial complex.

        Each parameter can be given as a list of lists of labels. Each simplex is kept (for sub_complex and complex)
        or erased (for relative) iff the set of labels of its vertices is contained in one of the lists of labels.
        For example sub_complex=[['blue', 'red'], ['blue', 'green']] chooses those simplices whose vertices are labeled
        either 'blue' and 'red' or 'blue' and 'green', but NOT 'red' and 'green'; also NOT all three colors,
        'blue','red','green'. The single-color simplices, e.g. all 'blue', ARE chosen.

        If the labels are single-digit integers, the parameters can also be given as lists of words separated by commas.
        For example `sub-complex = 01,02` will return complex with the sub-complex being all simplices with
        colors 0, 1, 01 or 02 -- in particular, no simplices with colors 12 or 012.

        Finally, the parameter can also be one of the following words:
            'all' ... allow all simplices
            'mono-chromatic' ... allow simplices containing only one label (='0,1,2' if labels are 0,1,2)
            'bi-chromatic' ... allow simplices containing at most two labels (='01,02,12' if labels are 0,1,2)
            'tri-chromatic' ... allow simplices containing at most three labels (='012'='all' if labels are 0,1,2)
            Can also be written as 'one-chromatic' or '1-chromatic' etc... The dash is optional.

        Keyword arguments:
        allow_unused_labels ... By default (False), an error is raised if the sub_complex/complex/relative parameters
                                contain a label that is not used. To suppress this behavior, set this parameter to True.
                                In that case the unused labels make no difference on the result.
        """
        return SimplicialComplex(self.core_alpha_complex.get_complex(sub_complex, complex, relative, allow_unused_labels))

    def radius_function(self):
        return {simplex : radius for simplex, radius in self.core_alpha_complex.simplicial_complex.simplex_weights.items()}

    def simplices(self):
        return set(self.core_alpha_complex.simplicial_complex.boundary.keys())

    def simplex_labels(self, simplex):
        return self.core_alpha_complex.simplex_labels_input(simplex)

    def simplex_radius(self, simplex):
        return self.core_alpha_complex.simplicial_complex.simplex_weights[simplex]

    def write(self) -> None:
        self.core_alpha_complex.write()
