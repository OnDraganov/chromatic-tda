from chromatic_tda.utils.singleton import singleton

@singleton
class SimplexUtils():
    def dimension(self, simplex) -> int:
        return len(simplex)-1
    