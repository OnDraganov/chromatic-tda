from abc import ABC

class TopologicalFunctions(ABC):
    pass

class TopologicalFunctions2D(TopologicalFunctions):
    pass

class TopologicalFunctions3D(TopologicalFunctions):
    pass

class TopologicalFunctionFactory():
    def getInstance(self, dimension: str) -> TopologicalFunctions:
        if dimension is "2D":
            return TopologicalFunctions2D()
        if dimension is "3D":
            return TopologicalFunctions3D()
        
        raise NotImplementedError
