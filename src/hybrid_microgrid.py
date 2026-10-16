# Extension
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from src.microgrid import MicroGrid


@dataclass
class HybridMicroGrid(MicroGrid):
    """
    Solar (x), battery (y), diesel (z) with three constraints:
        3x + 2y + 1z = D1
        4x +  y + 0z = D2
        1x + 1y + 1z = D3   (e.g. total kWh cap / emissions cap)
    """

    diesel_tariff: float = 900.0  # UGX / kWh (illustrative)
    coefficient_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [[3.0, 2.0, 1.0], [4.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
    )

    def __post_init__(self) -> None:
        self.coefficient_matrix = np.asarray(self.coefficient_matrix, dtype=float)
        if self.coefficient_matrix.shape != (3, 3):
            raise ValueError("Hybrid coefficient_matrix must be 3x3")

    def solve_day(self, d1: float, d2: float, d3: float) -> tuple[float, float, float]:
        rhs = np.array([d1, d2, d3], dtype=float)
        xyz = solve(self.coefficient_matrix, rhs)
        return tuple(float(v) for v in xyz)  # type: ignore[return-value]

    def daily_cost(self, solar_kwh: float, battery_kwh: float, diesel_kwh: float = 0.0) -> float:
        return (
            solar_kwh * self.solar_tariff
            + battery_kwh * self.battery_tariff
            + diesel_kwh * self.diesel_tariff
        )

n = 1000
base_d1, base_d2 = 38.0, 22.0
rng = np.random.default_rng(42)
mg  = MicroGrid()
pert = rng.uniform(-0.05, 0.05, size=(n, 2))
d1s = base_d1 * (1 + pert[:, 0])
d2s = base_d2 * (1 + pert[:, 1])
demands_mc = np.column_stack([d1s, d2s])
sols_mc = mg.solve_many_vectorised(demands_mc)

print("x std / mean:", sols_mc[:, 0].std() / sols_mc[:, 0].mean())
print("y std / mean:", sols_mc[:, 1].std() / sols_mc[:, 1].mean())
