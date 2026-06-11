from dataclasses import dataclass
import numpy as np
from typing import Callable


@dataclass
class ReactionSimulation:
    g: np.ndarray
    t: np.ndarray
    kinetics: Callable = None
    k1: float = 2.0
    k2: float = 1.0

    def __post_init__(
        self
    ):
        if self.kinetics is None:
            self.kinetics = self._default_kinetics

    def _default_kinetics(
        self,
        t
    ):
        a = np.exp(-self.k1*t)
        b = np.multiply(np.divide(self.k1, np.subtract(
            self.k2, self.k1)), np.subtract(a, np.exp(-self.k2*t)))
        c = 1 - b - a
        return np.column_stack([a, b, c])  # kinetics equations

    def simulation(
        self
    ):
        C = self.kinetics(self.t)  # G matrix
        if C.shape[1] != self.g.shape[1]:
            raise ValueError(
                f"The number of the phases in the kinetics (concentrations) matrix, G -- {C.shape[1]}"
                f"does not correspond to the number of the compounds in the structures matrix, F -- {self.g.shape[1]}"
            )
        X = C @ np.transpose(self.g)
        return X, C
