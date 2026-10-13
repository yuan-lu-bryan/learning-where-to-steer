"""Numerical checks for the CRD/Cone consensus solver; no datasets or GPU needed."""

import unittest

import numpy as np
import torch
from scipy.optimize import minimize

from experiments.crd_solver import solve_simplex_qp


class SimplexQPTests(unittest.TestCase):
    def test_two_objective_closed_form(self):
        matrices = torch.tensor(
            [
                [[2.0, 0.5], [0.5, 4.0]],
                [[1.0, 2.0], [2.0, 5.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ],
            dtype=torch.float64,
        )
        weights = solve_simplex_qp(matrices)
        for matrix, p in zip(matrices.numpy(), weights.numpy()):
            denominator = matrix[0, 0] + matrix[1, 1] - 2 * matrix[0, 1]
            if denominator > 0:
                p0 = np.clip((matrix[1, 1] - matrix[0, 1]) / denominator, 0.0, 1.0)
                np.testing.assert_allclose(p, [p0, 1 - p0], atol=1e-9)
            else:
                self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_objective_matches_independent_optimizer(self):
        rng = np.random.RandomState(13)
        for objectives in (3, 5):
            factors = rng.normal(size=(3, objectives, objectives + 2))
            matrices = np.einsum("bij,bkj->bik", factors, factors)
            weights = solve_simplex_qp(
                torch.tensor(matrices, dtype=torch.float64)
            ).numpy()
            for matrix, p in zip(matrices, weights):
                reference = minimize(
                    lambda q: q @ matrix @ q,
                    np.full(objectives, 1 / objectives),
                    jac=lambda q: 2 * matrix @ q,
                    bounds=[(0.0, 1.0)] * objectives,
                    constraints=[
                        {
                            "type": "eq",
                            "fun": lambda q: q.sum() - 1.0,
                            "jac": lambda q: np.ones_like(q),
                        }
                    ],
                    method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
                self.assertTrue(reference.success, reference.message)
                self.assertAlmostEqual(p @ matrix @ p, reference.fun, delta=1e-8)

    def test_zero_and_singular_inputs_remain_feasible(self):
        matrices = torch.stack((torch.zeros(5, 5), torch.ones(5, 5))).double()
        weights = solve_simplex_qp(matrices)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=torch.float64))

    def test_iterative_branch_matches_diagonal_optimum(self):
        diagonal = torch.arange(1.0, 9.0, dtype=torch.float64)
        matrix = torch.diag(diagonal).unsqueeze(0)
        weights = solve_simplex_qp(matrix)[0]
        expected = diagonal.reciprocal() / diagonal.reciprocal().sum()
        torch.testing.assert_close(weights, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
