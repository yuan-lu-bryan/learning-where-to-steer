"""Batched simplex QP solver used by CRD consensus and Cone fallback.

Solves min p^T Q p subject to p >= 0 and sum(p) = 1. The normalized
Gram matrix, including the samplers' existing Tikhonov term, is supplied by
the caller; this solver adds no further regularization. Small problems use
batched face enumeration. Larger problems use fixed-budget projected
optimization and active-set refinement. ``return_info=True`` exposes the
feasibility and gap checks for callers that need convergence diagnostics.
"""

import itertools
import math
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _supports(m, k, device_type, device_index):
    device = (
        torch.device(device_type, device_index)
        if device_index is not None
        else torch.device(device_type)
    )
    return torch.tensor(
        list(itertools.combinations(range(m), k)), dtype=torch.long, device=device
    )


def _project_simplex(x):
    """Euclidean projection row by row, with no cross-row reductions."""
    sorted_x = torch.sort(x, dim=-1, descending=True).values
    ranks = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    cumulative = sorted_x.cumsum(-1) - 1.0
    active = sorted_x - cumulative / ranks > 0
    rho = active.sum(-1).clamp_min(1)
    theta = cumulative.gather(-1, (rho - 1).unsqueeze(-1)) / rho.unsqueeze(-1)
    p = (x - theta).clamp_min(0.0)
    return p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)


def _objective(Q, p):
    return (p * torch.matmul(Q, p.unsqueeze(-1)).squeeze(-1)).sum(-1)


def _choose(Q, best, candidate, valid=None):
    if valid is None:
        valid = torch.isfinite(candidate).all(-1)
    improve = valid & (_objective(Q, candidate) < _objective(Q, best))
    return torch.where(improve.unsqueeze(-1), candidate, best)


def _equality_solve(A, rhs):
    solution, info = torch.linalg.solve_ex(A, rhs, check_errors=False)
    residual = (
        (torch.matmul(A, solution.unsqueeze(-1)).squeeze(-1) - rhs).abs().amax(-1)
    )
    valid = (info == 0) & torch.isfinite(solution).all(-1) & (residual <= 2e-8)
    return solution, valid


def _enumerate_faces(Q, best):
    B, m, _ = Q.shape
    rejected = torch.zeros(B, dtype=torch.long, device=Q.device)
    for k in range(2, m + 1):
        indices = _supports(m, k, Q.device.type, Q.device.index)
        n_faces = indices.shape[0]
        Q_face = Q[:, indices[:, :, None], indices[:, None, :]]
        system = Q.new_zeros((B, n_faces, k + 1, k + 1))
        system[..., :k, :k] = Q_face
        system[..., :k, k] = 1.0
        system[..., k, :k] = 1.0
        rhs = Q.new_zeros((B, n_faces, k + 1))
        rhs[..., k] = 1.0
        solved, valid = _equality_solve(system, rhs)
        rejected = rejected + (~valid).sum(-1)
        weights = solved[..., :k]
        valid = valid & (weights >= -1e-10).all(-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(
            0.0
        )
        mass = weights.sum(-1, keepdim=True)
        valid = valid & (mass.squeeze(-1) > 0)
        weights = weights / mass.clamp_min(torch.finfo(Q.dtype).tiny)
        values = (
            weights * torch.matmul(Q_face, weights.unsqueeze(-1)).squeeze(-1)
        ).sum(-1)
        values = torch.where(valid, values, torch.full_like(values, float("inf")))
        best_face = values.argmin(-1)
        row = torch.arange(B, device=Q.device)
        candidate = torch.zeros_like(best)
        candidate.scatter_(1, indices[best_face], weights[row, best_face])
        best = _choose(Q, best, candidate, valid[row, best_face])
    return best, rejected


def _polish_face(Q, p):
    B, m, _ = Q.shape
    support = (p > 1e-10).to(Q.dtype)
    system = Q.new_zeros((B, m + 1, m + 1))
    system[:, :m, :m] = Q * support.unsqueeze(-1) * support.unsqueeze(-2)
    system[:, :m, :m] += torch.diag_embed(1.0 - support)
    system[:, :m, m] = support
    system[:, m, :m] = support
    rhs = Q.new_zeros((B, m + 1))
    rhs[:, m] = 1.0
    solved, valid = _equality_solve(system, rhs)
    candidate = solved[:, :m] * support
    valid = valid & (candidate >= -1e-10).all(-1)
    candidate = torch.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(
        0.0
    )
    mass = candidate.sum(-1, keepdim=True)
    valid = valid & (mass.squeeze(-1) > 0)
    candidate = candidate / mass.clamp_min(torch.finfo(Q.dtype).tiny)
    return _choose(Q, p, candidate, valid), ~valid


def _projected_gradient(Q, initial, max_iter, refine_iter):
    # Gradient of (1/2)p^T Q p is Qp, with row-sum spectral upper bound.
    lipschitz = Q.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-30)
    p, momentum, best = initial, initial, initial
    acceleration = 1.0
    rejected = torch.zeros(Q.shape[0], dtype=torch.long, device=Q.device)
    for iteration in range(max_iter):
        gradient = torch.matmul(Q, momentum.unsqueeze(-1)).squeeze(-1)
        next_p = _project_simplex(momentum - gradient / lipschitz)
        best = _choose(Q, best, next_p)
        next_acceleration = (
            1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)
        ) / 2.0
        momentum = next_p + ((acceleration - 1.0) / next_acceleration) * (next_p - p)
        p, acceleration = next_p, next_acceleration
        if (iteration + 1) % 32 == 0 or iteration + 1 == max_iter:
            # This branch depends only on a Python loop index, not GPU state.
            best, invalid = _polish_face(Q, best)
            rejected = rejected + invalid.to(torch.long)
    best, active_rejects = _active_set_refine(Q, best, iterations=refine_iter)
    return best, rejected + active_rejects


@torch.no_grad()
def _active_set_refine(Q, p, iterations=64):
    """Refine feasible ``p`` for symmetric ``Q`` and return (p, rejects).

    Q should have dtype float64 and preferably per-row maximum absolute entry
    near one.  ``iterations`` is a fixed budget, not a convergence guarantee.
    The caller should check its usual simplex gap on the returned solution.
    ``rejects`` counts invalid KKT systems separately for each batch row.
    """
    B, m, _ = Q.shape
    # Exactly identical columns describe interchangeable simplex coordinates.
    # Move their mass to the first representative, preserving Qp and p^T Qp.
    # Keep them inactive in subsequent projections so a singular duplicate face
    # is not immediately recreated. Equality is exact: no near-equal columns
    # are merged and no regularization changes the supplied quadratic.
    coordinates = torch.arange(m, device=Q.device).expand(B, -1)
    representative = coordinates.clone()
    for j in range(1, m):
        equal = (Q[:, :, :j] == Q[:, :, j : j + 1]).all(dim=1)
        representative[:, j] = torch.where(
            equal.any(-1), equal.to(torch.long).argmax(-1), representative[:, j]
        )
    # Avoid scatter_add atomics: repeated identical inputs must remain
    # reproducible on CUDA for the paired benchmark's timing repeats.
    p = torch.stack(
        [torch.where(representative == j, p, 0.0).sum(-1) for j in range(m)], dim=-1
    )
    allowed = representative == coordinates
    support = (p > 0) & allowed
    rejects = torch.zeros(B, dtype=torch.long, device=Q.device)
    epsilon = torch.finfo(Q.dtype).eps
    tiny = torch.finfo(Q.dtype).tiny
    lipschitz = Q.abs().sum(-1).amax(-1, keepdim=True).clamp_min(tiny)
    rhs = Q.new_zeros((B, m + 1))
    rhs[:, m] = 1.0
    row = torch.arange(B, device=Q.device)

    def value(x):
        return (x * torch.matmul(Q, x.unsqueeze(-1)).squeeze(-1)).sum(-1)

    best = p.clone()
    best_value = value(best)
    best_g = torch.matmul(Q, best.unsqueeze(-1)).squeeze(-1)
    best_gap = (best_value - best_g.amin(-1)).clamp_min(0)
    for _ in range(int(iterations)):
        free = support.to(Q.dtype)
        system = Q.new_zeros((B, m + 1, m + 1))
        system[:, :m, :m] = Q * free.unsqueeze(-1) * free.unsqueeze(-2)
        system[:, :m, :m] += torch.diag_embed(1.0 - free)
        system[:, :m, m] = free
        system[:, m, :m] = free
        solved, info = torch.linalg.solve_ex(system, rhs, check_errors=False)
        residual = (
            (torch.matmul(system, solved.unsqueeze(-1)).squeeze(-1) - rhs)
            .abs()
            .amax(-1)
        )
        valid = (info == 0) & torch.isfinite(solved).all(-1) & (residual <= 2e-8)
        rejects += (~valid).to(torch.long)
        candidate = (
            torch.nan_to_num(solved[:, :m], nan=0.0, posinf=0.0, neginf=0.0) * free
        )
        feasible_face = (candidate >= -1e-12).all(-1)

        direction = candidate - p
        decreasing = direction < 0
        distance = torch.where(
            decreasing,
            p / (-direction).clamp_min(tiny),
            torch.full_like(p, float("inf")),
        )
        step = distance.amin(-1).clamp(0.0, 1.0)
        boundary = p + step.unsqueeze(-1) * direction
        face_point = torch.where(
            feasible_face.unsqueeze(-1), candidate, boundary
        ).clamp_min(0)
        face_point /= face_point.sum(-1, keepdim=True).clamp_min(tiny)

        # A singular system only affects its own row, which retains a feasible
        # descent update over the canonical coordinates.
        old_gradient = torch.matmul(Q, p.unsqueeze(-1)).squeeze(-1)
        fallback = _project_simplex(
            (p - old_gradient / lipschitz).masked_fill(~allowed, float("-inf"))
        )
        next_p = torch.where(valid.unsqueeze(-1), face_point, fallback)
        gradient = torch.matmul(Q, next_p.unsqueeze(-1)).squeeze(-1)
        next_value = (next_p * gradient).sum(-1)
        gap = (next_value - gradient.amin(-1)).clamp_min(0)

        # At a face optimum, include the smallest reduced-cost coordinate. At
        # an infeasible face minimizer, drop the constraints hit by the step.
        next_support = support & ~(decreasing & (face_point <= 32.0 * epsilon))
        inactive_gradient = gradient.masked_fill(support | ~allowed, float("inf"))
        entrant = inactive_gradient.argmin(-1)
        violation = inactive_gradient[row, entrant] < next_value - 64.0 * epsilon
        add = torch.zeros_like(support)
        add.scatter_(1, entrant.unsqueeze(-1), violation.unsqueeze(-1))
        face_support = (face_point > 0) | add
        next_support = torch.where(
            feasible_face.unsqueeze(-1), face_support, next_support
        )
        support = torch.where(valid.unsqueeze(-1), next_support, next_p > 0)
        p = next_p

        # Keep the best objective; use gap to break machine-precision ties.
        close = (next_value - best_value).abs() <= 64.0 * epsilon
        improve = (next_value < best_value) | (close & (gap < best_gap))
        best = torch.where(improve.unsqueeze(-1), p, best)
        best_value = torch.where(improve, next_value, best_value)
        best_gap = torch.where(improve, gap, best_gap)

    return best, rejects


@torch.no_grad()
def solve_simplex_qp(
    Q, *, enum_max_m=6, max_iter=256, tol=1e-7, refine_iter=64, return_info=False
):
    """Return a feasible simplex vector for every batch row and optional checks.

    ``enum_max_m`` selects the small-m algorithm; it is NOT a supported-m limit.
    Above that threshold FISTA supports any positive m within memory/time limits.
    ``gap`` is 2*(p^T Qp - min(Qp)); for PSD Q and feasible p it upper bounds
    objective suboptimality. Large-m fixed-budget solves may be uncertified;
    callers must inspect info['converged'] rather than assume exactness.
    Mass error is checked on the returned dtype. The gap is evaluated on its
    float64-renormalized interpretation, so rounding of total mass does not
    masquerade as an optimization residual (or conceal one).
    """
    if Q.ndim != 3 or Q.shape[-1] != Q.shape[-2] or Q.shape[0] < 1 or Q.shape[-1] < 1:
        raise ValueError("Q must have shape (nonempty_batch, positive_m, positive_m)")
    if Q.dtype not in (torch.float32, torch.float64):
        raise TypeError("Q must be float32 or float64")
    if (
        enum_max_m < 0
        or enum_max_m > 8
        or max_iter < 1
        or refine_iter < 0
        or not math.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError("enum_max_m must be 0..8; max_iter/tol must be positive")
    if not hasattr(torch.linalg, "solve_ex"):
        raise RuntimeError(
            "This solver requires torch.linalg.solve_ex(check_errors=False)"
        )
    B, m, _ = Q.shape
    original = 0.5 * (Q.to(torch.float64) + Q.to(torch.float64).transpose(-1, -2))
    finite_input = torch.isfinite(original).all(-1).all(-1)
    safe = torch.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
    scale = safe.abs().amax(-1).amax(-1).clamp_min(1e-300)
    normalized = safe / scale[:, None, None]
    best = Q.new_full((B, m), 1.0 / m, dtype=torch.float64)
    vertex = torch.zeros_like(best)
    vertex.scatter_(
        1, normalized.diagonal(dim1=-2, dim2=-1).argmin(-1, keepdim=True), 1.0
    )
    best = _choose(normalized, best, vertex)
    if m <= enum_max_m:
        best, rejected = _enumerate_faces(normalized, best)
        algorithm, iterations = "grouped_kkt_enumeration", 0
    else:
        active_budget = max(int(refine_iter), 4 * m) if refine_iter else 0
        best, rejected = _projected_gradient(
            normalized, best, int(max_iter), active_budget
        )
        algorithm, iterations = "fista_primal_active_set", int(max_iter) + active_budget
    result = best.to(Q.dtype)
    result = result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(Q.dtype).tiny)
    actual = result.to(torch.float64)
    mass = actual.sum(-1, keepdim=True)
    certificate_pi = actual / mass.clamp_min(torch.finfo(torch.float64).tiny)
    Qp = torch.matmul(safe, certificate_pi.unsqueeze(-1)).squeeze(-1)
    objective = (certificate_pi * Qp).sum(-1)
    gap = (2.0 * (objective - Qp.amin(-1))).clamp_min(0.0)
    mass_error = (mass.squeeze(-1) - 1.0).abs()
    nonnegative = (actual >= 0).all(-1)
    tolerance = float(tol) * scale.clamp_min(1.0)
    converged = (
        finite_input
        & nonnegative
        & (mass_error <= 5e-7)
        & torch.isfinite(gap)
        & (gap <= tolerance)
    )
    info = {
        "gap": gap,
        "objective": objective,
        "converged": converged,
        "mass_error": mass_error,
        "finite_input": finite_input,
        "tolerance": tolerance,
        "scale": scale,
        "iterations": iterations,
        "algorithm": algorithm,
        "refine_iterations": active_budget if m > enum_max_m else 0,
        "certificate_point": "returned_pi_normalized_in_float64",
        "rejected_face_systems": rejected,
    }
    return (result, info) if return_info else result
