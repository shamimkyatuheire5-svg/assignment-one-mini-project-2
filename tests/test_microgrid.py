import numpy as np
import pytest
from src.microgrid import MicroGrid


def test_solve_day_matches_hand_calculation():
    mg = MicroGrid()
    # 3x + 2y = 8 ; 4x + y = 7  ->  x = 1.2, y = 2.2
    x, y = mg.solve_day(8, 7)
    assert x == pytest.approx(1.2)
    assert y == pytest.approx(2.2)


def test_loop_and_vectorised_agree():
    mg = MicroGrid()
    demands = np.array([[40.0, 22.0], [30.0, 18.0], [38.0, 23.0]])
    a = mg.solve_many_loop(demands)
    b = mg.solve_many_vectorised(demands)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_negative_demand_rejected():
    mg = MicroGrid()
    with pytest.raises(ValueError):
        mg.solve_day(-1, 5)


def test_singular_matrix_not_well_posed():
    mg = MicroGrid(coefficient_matrix=np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert not mg.is_well_posed()


def test_empty_demand_array():
    mg = MicroGrid()
    empty = np.empty((0, 2))
    out = mg.solve_many_vectorised(empty)
    assert out.shape == (0, 2)


def test_nnls_handles_infeasible_day():
    mg = MicroGrid()
    # Pick a demand that gives negative battery usage
    x, y = mg.solve_day(1, 20)
    assert y < 0  # confirm infeasible
    x2, y2 = mg.solve_nonnegative(1, 20)
    assert x2 >= 0 and y2 >= 0