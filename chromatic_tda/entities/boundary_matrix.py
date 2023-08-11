
from chromatic_tda.utils.topological_functions import TopologicalFunctions, TopologicalFunctionFactory

class BoundaryMatrix():
    def __init__(self):
        self.dimension = "2D"
        self.function_controller : TopologicalFunctions = TopologicalFunctionFactory().getInstance(self.dimension)
