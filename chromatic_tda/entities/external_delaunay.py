from abc import ABC, abstractmethod
from chromatic_tda.entities.simplicial_complex import SimplicialComplex

class ExternalDelaunay(ABC):
    pass

class ScipyExternalDelaunay(ExternalDelaunay):
    pass

class DelaunayMediator(ABC):
    @abstractmethod
    def convert(self, external_delaunay: ExternalDelaunay) -> SimplicialComplex:
        pass

class ScipyDelaunayMediator(DelaunayMediator):
    def convert(self, external_delaunay) -> SimplicialComplex:
        raise NotImplementedError
