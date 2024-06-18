import numpy as np
from dataclasses import dataclass


@dataclass
class StackOfSpheres:
    center: np.array
    radii: dict[any, float]
    maximum_radius: float
    maximum_labels: set

    def __init__(self, center: np.array, radii: dict[any, float]):
        self.center = center
        self.radii = radii
        self.maximum_radius = max(radii.values())
        self.maximum_labels = {label for label, radius in self.radii.items() if np.isclose(radius, self.maximum_radius)}
