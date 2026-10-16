"""
MicroGrid dispatch planner.

Solves the 2x2 linear system:
    3x + 2y = D1   (daytime load, kWh)
    4x + y  = D2   (critical-equipment load, kWh)

where x = solar energy drawn, y = battery energy drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import nnls

# Illustrative tariffs (UGX / kWh)
SOLAR_TARIFF = 150.0
BATTERY_TARIFF = 450.0


@dataclass
class MicroGrid:
    """A solar + battery micro-grid with two linear demand constraints."""

    coefficient_matrix: np.ndarray = field(
        default_factory=lambda: np.array([[3.0, 2.0], [4.0, 1.0]])
    )
    solar_tariff: float = SOLAR_TARIFF
    battery_tariff: float = BATTERY_TARIFF

    def __post_init__(self) -> None:
        self.coefficient_matrix = np.asarray(self.coefficient_matrix, dtype=float)
        if self.coefficient_matrix.shape != (2, 2):
            raise ValueError("coefficient_matrix must be 2x2")

    # ---------- diagnostics ----------
    @property
    def determinant(self) -> float:
        """Determinant of the coefficient matrix."""
        return float(np.linalg.det(self.coefficient_matrix))

    @property
    def condition_number(self) -> float:
        """Condition number (2-norm) of the coefficient matrix."""
        return float(np.linalg.cond(self.coefficient_matrix))

    def is_well_posed(self, tol: float = 1e-8) -> bool:
        """A system is well-posed if det != 0 and cond is finite."""
        return abs(self.determinant) > tol and np.isfinite(self.condition_number)

    # ---------- solving ----------
    def solve_day(self, d1: float, d2: float) -> tuple[float, float]:
        """Solve one day's 2x2 system. Returns (solar_kwh, battery_kwh)."""
        if d1 < 0 or d2 < 0:
            raise ValueError("Demand values must be non-negative.")
        rhs = np.array([d1, d2], dtype=float)
        x, y = solve(self.coefficient_matrix, rhs)
        return float(x), float(y)

    def solve_many_loop(self, demands: np.ndarray) -> np.ndarray:
        """Solve many days in an explicit Python loop. demands shape (n, 2)."""
        out = np.empty_like(demands, dtype=float)
        for i, (d1, d2) in enumerate(demands):
            out[i, 0], out[i, 1] = self.solve_day(d1, d2)
        return out

    def solve_many_vectorised(self, demands: np.ndarray) -> np.ndarray:
        """Solve all days in one vectorised call. demands shape (n, 2)."""
        rhs = np.asarray(demands, dtype=float).T  # shape (2, n)
        sol = solve(self.coefficient_matrix, rhs)  # shape (2, n)
        return sol.T  # back to (n, 2)

    # ---------- feasibility ----------
    def flag_infeasible(self, solutions: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Return boolean mask: True where any component is negative."""
        return np.any(solutions < -tol, axis=1)

    def solve_nonnegative(self, d1: float, d2: float) -> tuple[float, float]:
        """NNLS fallback: minimise ||Ax - b|| with x >= 0."""
        rhs = np.array([d1, d2], dtype=float)
        x, _ = nnls(self.coefficient_matrix, rhs)
        return float(x[0]), float(x[1])

    def handle_infeasible(
        self, demands: np.ndarray, strategy: str = "nnls"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (solutions, infeasible_mask).

        strategy='clip'  -> clip negatives to zero (crude but transparent)
        strategy='nnls'  -> re-solve infeasible days with non-negative least squares
        """
        sols = self.solve_many_vectorised(demands)
        mask = self.flag_infeasible(sols)
        if strategy == "clip":
            sols = np.clip(sols, 0.0, None)
        elif strategy == "nnls":
            for i in np.where(mask)[0]:
                sols[i] = self.solve_nonnegative(*demands[i])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return sols, mask

    # ---------- cost ----------
    def daily_cost(self, solar_kwh: float, battery_kwh: float) -> float:
        """Daily energy cost in UGX."""
        return solar_kwh * self.solar_tariff + battery_kwh * self.battery_tariff

    def monthly_cost(self, daily_costs: Iterable[float], days: int = 30) -> float:
        """Extrapolate a mean daily cost to a 30-day month."""
        arr = np.asarray(list(daily_costs), dtype=float)
        if arr.size == 0:
            raise ValueError("daily_costs is empty")
        return float(arr.mean() * days)

    # ---------- CSV I/O ----------
    @staticmethod
    def generate_demand_csv(
        path: str | Path,
        days: int = 30,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create a synthetic 30-day demand file with a weekly pattern + noise.
        Daytime load peaks mid-week; critical load is flatter.
        """
        rng = np.random.default_rng(seed)
        dow = np.arange(days) % 7  # 0=Mon ... 6=Sun
        # Weekly pattern (kWh): busier Mon–Fri, quieter weekend
        pattern_d1 = np.array([38, 40, 39, 41, 37, 30, 28], dtype=float)
        pattern_d2 = np.array([22, 23, 22, 24, 22, 19, 18], dtype=float)
        d1 = pattern_d1[dow] + rng.normal(0, 1.5, days)
        d2 = pattern_d2[dow] + rng.normal(0, 1.0, days)
        df = pd.DataFrame(
            {
                "day": np.arange(1, days + 1),
                "dow": dow,
                "D1_daytime_kwh": np.round(np.clip(d1, 0, None), 2),
                "D2_critical_kwh": np.round(np.clip(d2, 0, None), 2),
            }
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df

    @staticmethod
    def load_demand_csv(path: str | Path) -> np.ndarray:
        """Load a demand CSV, returning an (n, 2) array of [D1, D2]."""
        df = pd.read_csv(path)
        return df[["D1_daytime_kwh", "D2_critical_kwh"]].to_numpy(dtype=float)

    # ---------- interactive ----------
    @staticmethod
    def _prompt_positive(label: str) -> float:
        while True:
            raw = input(f"{label}: ").strip()
            if raw == "":
                print("  Empty input — try again.")
                continue
            try:
                value = float(raw)
            except ValueError:
                print("  Not a number — try again.")
                continue
            if value < 0:
                print("  Must be non-negative — try again.")
                continue
            return value

    def interactive_solve(self) -> tuple[float, float]:
        """Prompt the user for D1 and D2 and solve."""
        d1 = self._prompt_positive("Daytime load D1 (kWh)")
        d2 = self._prompt_positive("Critical load D2 (kWh)")
        return self.solve_day(d1, d2)

    # ---------- dunder ----------
    def __repr__(self) -> str:
        return (
            f"MicroGrid(A={self.coefficient_matrix.tolist()}, "
            f"det={self.determinant:.3f}, cond={self.condition_number:.3f})"
        )
df = pd.read_csv("data/demand_30days.csv")
print(df.shape)         # (30, 4)
print(df.dtypes)        # int64, int64, float64, float64
print(df.describe())    # means should be ~ 38 (D1) and ~ 22 (D2)
print(df.isna().sum())  # all zeros