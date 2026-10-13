import math
import numpy as np
import os
from time import time, sleep
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.off_moo_bench.utils import get_N_nondominated_indices
from offline_moo.utils import get_quantile_solutions, spread
from experiments.paretoflow_utils import out_of_bounds_ratio_in_normalized_space
from experiments.paretoflow_utils import empirical_pareto_front

# Optional: use your existing Das–Dennis generator if available
try:
    from experiments.paretoflow_utils import get_reference_directions
except Exception:
    get_reference_directions = None


def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def _sqrtm_psd(
    M: torch.Tensor,
    eps: float = 1e-8,
    jitter: float = 1e-6,
    max_tries: int = 8,
    svd_on_cpu: bool = True,
) -> torch.Tensor:
    """
    Return S such that S S^T ≈ (M + jitter*I) (PSD factor).
    Tries Cholesky with increasing jitter; falls back to CPU SVD (float64).
    """
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = _sym(M)

    D = M.shape[0]
    I = torch.eye(D, device=M.device, dtype=M.dtype)

    j = float(jitter)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(M + j * I)
        if int(info.max().item()) == 0:
            return L
        j *= 10.0

    # fallback
    if svd_on_cpu:
        Mcpu = (M + j * I).detach().to("cpu", dtype=torch.float64)
        U, S, _ = torch.linalg.svd(Mcpu, full_matrices=True)
        S = torch.clamp(S, min=eps)
        Scpu = U @ torch.diag(torch.sqrt(S))
        return Scpu.to(device=M.device, dtype=M.dtype)

    U, S, _ = torch.linalg.svd(M + j * I, full_matrices=False)
    S = torch.clamp(S, min=eps)
    return U @ torch.diag(torch.sqrt(S))


@torch.no_grad()
def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    den = torch.sqrt((a * a).sum() * (b * b).sum()) + eps
    return num / den


@torch.no_grad()
def _orient_eigenvectors_by_label(
    U: torch.Tensor,
    X_feat: torch.Tensor,
    y_w: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Orient eigenvectors so that corr(X_feat @ u_j, y_w) is nonnegative.

    This matters for noise steering because u_j and -u_j define opposite
    perturbation directions. For projectors U U^T, the sign is irrelevant,
    but using a consistent convention is harmless.
    """
    U = U.clone()
    y_w = y_w.view(-1)

    corr_list = []

    for j in range(U.shape[1]):
        uj = U[:, j]
        proj = (X_feat @ uj).view(-1)
        rho = pearson_corr(proj, y_w, eps=eps)

        if float(rho) < 0.0:
            U[:, j] = -uj
            rho = -rho

        corr_list.append(float(rho.detach().cpu()))

    U = U / (U.norm(dim=0, keepdim=True) + eps)

    return U, corr_list


@torch.no_grad()
def reflect_to_box(x: torch.Tensor, xl, xu, eps: float = 1e-12) -> torch.Tensor:
    """
    Coordinate-wise reflection map R^d -> [xl, xu]^d (z-scored normalized space).

    For each dimension with width w = xu - xl:
      r = (x - xl) mod (2w) in [0, 2w)
      y = r              if r <= w
          2w - r         otherwise
      out = xl + y

    This avoids boundary pile-up (unlike clamp) and guarantees feasibility.
    """
    if xl is None or xu is None:
        return x

    xl_t = torch.as_tensor(xl, device=x.device, dtype=x.dtype)
    xu_t = torch.as_tensor(xu, device=x.device, dtype=x.dtype)

    if xl_t.ndim == 1:
        xl_t = xl_t.view(1, -1)
    if xu_t.ndim == 1:
        xu_t = xu_t.view(1, -1)

    # broadcast to (N, d)
    if xl_t.shape[0] == 1 and x.shape[0] != 1:
        xl_t = xl_t.expand(x.shape[0], -1)
    if xu_t.shape[0] == 1 and x.shape[0] != 1:
        xu_t = xu_t.expand(x.shape[0], -1)

    w = xu_t - xl_t
    w_safe = torch.clamp(w, min=eps)

    two_w = 2.0 * w_safe
    r = torch.remainder(x - xl_t, two_w)  # in [0, 2w)
    y = torch.where(r <= w_safe, r, two_w - r)
    out = xl_t + y

    # Safety: numerical edge cases
    out = torch.min(torch.max(out, xl_t), xu_t)
    return out


@torch.no_grad()
def laplace_kernel_self_with_metric(
    Z: torch.Tensor,
    M: torch.Tensor,
    L: float,
    eps: float = 1e-12,
    jitter: float = 1e-6,
):
    """
    Self-kernel K(Z,Z) for Laplace Mahalanobis kernel with metric M.
    Returns:
      K:   (n,n)
      dist:(n,n) where dist_ij = sqrt((zi-zj)^T M_eff (zi-zj))
      M_eff: (D,D) implicit metric actually used (S S^T) after stabilization
    """
    S = _sqrtm_psd(M, jitter=jitter)
    M_eff = S @ S.t()
    Zs = Z @ S
    dist = torch.cdist(Zs, Zs)
    K = torch.exp(-(dist / float(L)))
    return K, dist, M_eff


def krr_solve_alpha(
    Kzz: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    max_tries: int = 8,
    jitter_initial: float = 1e-7,
    jitter_growth: float = 10.0,
    verbose: bool = True,
):
    """
    alpha = (K + lam I)^{-1} y

    If the original Cholesky solve succeeds, this behaves exactly like the
    original implementation.

    If the original Cholesky fails, use a stable fallback with:
      - NaN/Inf cleanup,
      - symmetrization,
      - adaptive diagonal jitter.
    """
    n = Kzz.shape[0]

    try:
        A = Kzz + lam * torch.eye(n, device=Kzz.device, dtype=Kzz.dtype)
        Lchol = torch.linalg.cholesky(A)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        alpha = torch.cholesky_solve(y, Lchol)

        return alpha.squeeze(1) if alpha.shape[1] == 1 else alpha

    except RuntimeError as original_error:
        if verbose:
            print(
                "[RFM/KRR] Original Cholesky failed; switching to stable "
                "fallback solver. "
                f"Original error: {str(original_error).splitlines()[0]}"
            )

    if Kzz.ndim != 2 or Kzz.shape[0] != Kzz.shape[1]:
        raise ValueError(f"Kzz must be square, got shape={tuple(Kzz.shape)}")

    n = Kzz.shape[0]
    device = Kzz.device
    dtype = Kzz.dtype

    if y.ndim == 1:
        y_solve = y.unsqueeze(1)
    elif y.ndim == 2:
        y_solve = y
    else:
        raise ValueError(
            f"y must have shape (n,), (n,1), or (n,m), got {tuple(y.shape)}"
        )

    if y_solve.shape[0] != n:
        raise ValueError(
            f"Kzz and y have incompatible leading dimensions: "
            f"Kzz.shape={tuple(Kzz.shape)}, y.shape={tuple(y_solve.shape)}"
        )

    y_solve = y_solve.to(device=device, dtype=dtype)

    K = torch.nan_to_num(Kzz, nan=0.0, posinf=0.0, neginf=0.0)
    K = 0.5 * (K + K.t())

    I = torch.eye(n, device=device, dtype=dtype)

    base_lam = float(lam)
    extra_jitter = 0.0

    for attempt in range(int(max_tries) + 1):
        reg = base_lam + extra_jitter

        A = K + reg * I
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = 0.5 * (A + A.t())

        Lchol, info = torch.linalg.cholesky_ex(A)

        if int(info.max().item()) == 0:
            if verbose:
                if extra_jitter > 0.0:
                    print(
                        "[RFM/KRR] Stable Cholesky succeeded with extra "
                        f"diagonal jitter={extra_jitter:.3e}; "
                        f"effective lambda={reg:.3e}."
                    )
                else:
                    print(
                        "[RFM/KRR] Stable Cholesky succeeded after "
                        "symmetrization without extra jitter."
                    )

            alpha = torch.cholesky_solve(y_solve, Lchol)

            return alpha.squeeze(1) if alpha.shape[1] == 1 else alpha

        if extra_jitter == 0.0:
            extra_jitter = float(jitter_initial)
        else:
            extra_jitter *= float(jitter_growth)

    final_extra = max(extra_jitter, 1e-3)
    final_reg = base_lam + final_extra

    A = K + final_reg * I
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.t())

    Lchol, info = torch.linalg.cholesky_ex(A)

    if int(info.max().item()) == 0:
        if verbose:
            print(
                "[RFM/KRR] Stable Cholesky succeeded with final aggressive "
                f"diagonal jitter={final_extra:.3e}; "
                f"effective lambda={final_reg:.3e}."
            )

        alpha = torch.cholesky_solve(y_solve, Lchol)

        return alpha.squeeze(1) if alpha.shape[1] == 1 else alpha

    raise RuntimeError(
        "[RFM/KRR] Stable Cholesky failed even after adaptive jitter. "
        f"base_lam={base_lam:.3e}, final_extra_jitter={final_extra:.3e}, "
        f"final_effective_lambda={final_reg:.3e}, "
        f"last_cholesky_info={int(info.max().item())}. "
        "Try increasing --rfm-lam, reducing --rfm-L, or reducing "
        "--rfm-train-samples."
    )


@torch.no_grad()
def _corner_preserving_maxmin_subsample(
    W_all: torch.Tensor,
    n_keep: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Deterministically select exactly n_keep rows from a Das-Dennis grid.

    This implements vertex-initialized farthest-first traversal:

        1. Preserve simplex corners e_1, ..., e_m whenever they exist.
        2. Repeatedly add the point whose nearest selected point is farthest away.

    Equivalently, this greedily reduces the covering radius

        max_{w in W_all} min_{s in selected} ||w - s||_2.

    Args:
        W_all:
            Tensor of shape (N, m). Rows should lie on the unit simplex.
        n_keep:
            Number of rows to keep.
        eps:
            Numerical stabilizer for row normalization.

    Returns:
        idx:
            LongTensor of shape (n_keep,) containing selected row indices.
    """
    if W_all.ndim != 2:
        raise ValueError(f"W_all must be 2D, got shape={tuple(W_all.shape)}")

    N, m = W_all.shape
    n_keep = int(n_keep)

    if n_keep <= 0:
        raise ValueError(f"n_keep must be positive, got {n_keep}")

    if n_keep >= N:
        return torch.arange(N, device=W_all.device, dtype=torch.long)

    # Work in normalized simplex coordinates defensively.
    W = W_all / (W_all.sum(dim=1, keepdim=True) + eps)

    selected = []
    selected_mask = torch.zeros(N, device=W.device, dtype=torch.bool)

    for j in range(m):
        if len(selected) >= n_keep:
            break

        corner = torch.zeros(m, device=W.device, dtype=W.dtype)
        corner[j] = 1.0

        d = torch.norm(W - corner.unsqueeze(0), dim=1)
        idx = int(torch.argmin(d).item())

        if not bool(selected_mask[idx]):
            selected.append(idx)
            selected_mask[idx] = True

    dist_to_selected = torch.full(
        (N,),
        float("inf"),
        device=W.device,
        dtype=W.dtype,
    )

    for idx in selected:
        d = torch.norm(W - W[idx].unsqueeze(0), dim=1)
        dist_to_selected = torch.minimum(dist_to_selected, d)

    # Do not select already selected points again.
    dist_to_selected[selected_mask] = -float("inf")

    while len(selected) < n_keep:
        idx = int(torch.argmax(dist_to_selected).item())

        selected.append(idx)
        selected_mask[idx] = True

        d = torch.norm(W - W[idx].unsqueeze(0), dim=1)
        dist_to_selected = torch.minimum(dist_to_selected, d)

        # Do not select already selected points again.
        dist_to_selected[selected_mask] = -float("inf")

    return torch.tensor(selected, device=W.device, dtype=torch.long)


@torch.no_grad()
def _make_weight_set(
    m: int,
    n_weights: int,
    weight_mode: str = "das-dennis",
    dirichlet_alpha: float = 1.0,
    dd_partitions: int = 0,
    device=None,
    dtype=torch.float32,
):
    """
    Returns W: (K,m), rows sum to 1.

    Modes:
      - "per_obj":
            one-hot weights, K=m.
      - "dirichlet":
            random Dirichlet(alpha) weights, K=n_weights.
      - "das-dennis":
            generate the smallest Das-Dennis grid with at least n_weights
            directions unless dd_partitions is explicitly provided.

            If the grid has more than n_weights directions, use
            deterministic corner-preserving max-min subsampling instead of
            random subsampling.
    """
    if device is None:
        device = torch.device("cpu")

    n_weights = int(n_weights)
    m = int(m)

    if n_weights <= 0:
        raise ValueError(f"n_weights must be positive, got {n_weights}")

    mode = weight_mode.lower()

    if mode in ["per_obj", "onehot", "one_hot"]:
        W = torch.eye(m, device=device, dtype=dtype)
        return W

    if mode in ["dirichlet", "dir"]:
        alpha = torch.full((m,), float(dirichlet_alpha), device=device, dtype=dtype)
        dist = torch.distributions.Dirichlet(alpha)
        W = dist.sample((n_weights,))
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        return W

    if mode in ["das-dennis", "das_dennis", "dd"]:
        if get_reference_directions is None:
            raise RuntimeError(
                "Das-Dennis weight_mode requested but "
                "get_reference_directions is not importable."
            )

        # Choose the smallest partition count whose full grid reaches n_weights.
        if dd_partitions is None or int(dd_partitions) <= 0:
            p = 1
            while math.comb(m + p - 1, p) < n_weights:
                p += 1
        else:
            p = int(dd_partitions)

        W_np = get_reference_directions("uniform", m, n_partitions=p)
        W_all = torch.tensor(W_np, device=device, dtype=dtype)

        if W_all.shape[0] > n_weights:
            idx = _corner_preserving_maxmin_subsample(
                W_all=W_all,
                n_keep=n_weights,
            )
            W = W_all[idx]
        else:
            W = W_all

        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        return W

    raise ValueError(f"Unknown weight_mode: {weight_mode}")


@torch.no_grad()
def _compute_grads_multioutput(
    Z: torch.Tensor,  # (n,D)
    Y: torch.Tensor,  # (n,m)
    M: torch.Tensor,  # (D,D)
    L: float,
    lam: float,
    eps: float = 1e-12,
    jitter: float = 1e-6,
):
    """
    One pass under metric M:
      - KRR: A = (K+lam I)^-1 Y  => (n,m)
      - gradients at training points: grads (n,m,D) where grads[i,a,:] = ∇ f_a(z_i)
    Returns:
      grads: (n,m,D)
      Kzz:   (n,n)
      dist:  (n,n)
    """
    Kzz, dist, M_eff = laplace_kernel_self_with_metric(Z, M, L, eps=eps, jitter=jitter)

    # multi-output coefficients
    A = krr_solve_alpha(Kzz, Y, lam=lam)  # (n,m)

    P = Kzz / (dist + eps)
    P.fill_diagonal_(0.0)

    s = P @ A

    WZ = torch.einsum("ij,jm,jd->imd", P, A, Z)

    # Use M_eff consistently with the distances/kernel (important when jittered)
    ZM = Z @ M_eff  # (n,D)
    WZ_M = (WZ.reshape(-1, Z.shape[1]) @ M_eff).reshape(
        Z.shape[0], A.shape[1], Z.shape[1]
    )

    grads = (WZ_M - s.unsqueeze(-1) * ZM.unsqueeze(1)) / float(L)  # (n,m,D)
    grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)

    return grads, Kzz, dist


@torch.no_grad()
def _shared_metric_rfm_recursion(
    Z: torch.Tensor,
    Y: torch.Tensor,
    W_metric: torch.Tensor,  # (K,m) weight set for covariance C
    L: float,
    lam: float,
    T: int,
    eps: float = 1e-12,
    normalize_M_trace: bool = False,
    metric_jitter: float = 1e-6,
):
    """
    Shared metric recursion:
      M_{t+1} = (1/n) Σ_i G_i C G_i^T, where C = E[w w^T], G_i is D×m gradients matrix.
    """
    n, D = Z.shape

    # covariance C = mean_k w_k w_k^T
    C = (W_metric.t() @ W_metric) / float(W_metric.shape[0])  # (m,m)
    C = _sym(C)

    M = torch.eye(D, device=Z.device, dtype=Z.dtype)

    for _ in range(int(T)):
        grads, _, _ = _compute_grads_multioutput(
            Z, Y, M, L=L, lam=lam, eps=eps, jitter=metric_jitter
        )  # (n,m,D)

        # G: (n,D,m)
        G = grads.transpose(1, 2).contiguous()

        # GC: (n,D,m)
        GC = torch.matmul(G, C)

        M_next = torch.einsum("ndm,nem->de", GC, G) / float(n)
        M_next = _sym(torch.nan_to_num(M_next, nan=0.0, posinf=0.0, neginf=0.0))

        if normalize_M_trace:
            tr = torch.trace(M_next)
            if float(tr) > 0:
                M_next = M_next * (D / tr)

        M = M_next

    return M


@torch.no_grad()
def all_eigenvectors_svd(
    M: torch.Tensor,
    jitter: float = 1e-6,
    eps: float = 1e-12,
    svd_on_cpu: bool = True,
):
    """
    Return ALL eigenvectors of symmetric PSD M as columns (D,D),
    together with eigenvalues sorted in descending order.

    For PSD matrices, singular values == eigenvalues and left singular vectors
    == eigenvectors.
    """
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = 0.5 * (M + M.t())
    D = M.shape[0]
    A = M + float(jitter) * torch.eye(D, device=M.device, dtype=M.dtype)

    if svd_on_cpu:
        A_cpu = A.detach().to("cpu", dtype=torch.float64)
        U_cpu, S_cpu, _ = torch.linalg.svd(A_cpu, full_matrices=True)
        U = U_cpu.to(device=M.device, dtype=M.dtype)
        S = S_cpu.to(device=M.device, dtype=M.dtype)
    else:
        U, S, _ = torch.linalg.svd(A, full_matrices=True)

    U = U / (U.norm(dim=0, keepdim=True) + eps)
    return U, S


@torch.no_grad()
def randomized_psd_eig_from_gw_gpu(
    g_w: torch.Tensor,
    q: int,
    niter: int = 1,
    eps: float = 1e-12,
    jitter: float = 0.0,
    oversample: int = 10,
    return_rank: int = None,
):
    """
    Randomized eigensolver for the PSD matrix

        M = (1/n) g_w.T @ g_w,

    without explicitly forming M.

    This stays on the device of g_w. For your RFM usage, g_w is on GPU,
    so this avoids both:
        - forming the dense D x D AGOP matrix in the low-rank branch;
        - calling torch.svd_lowrank on either GPU or CPU.

    Args:
        g_w:
            Tensor of shape (n, D).
        q:
            Requested returned rank.
        niter:
            Number of power/subspace iterations.
        eps:
            Numerical stabilizer.
        jitter:
            Optional diagonal shift applied in the small projected matrix.
            This shifts eigenvalues but does not change the target eigenspace
            in exact arithmetic.
        oversample:
            Extra sketch dimensions for randomized approximation quality.
        return_rank:
            Number of eigenpairs to return. If None, return q.

    Returns:
        U:
            Tensor of shape (D, return_rank), approximate leading eigenvectors.
        evals:
            Tensor of shape (return_rank,), approximate leading eigenvalues.
    """
    if g_w.ndim != 2:
        raise ValueError(f"g_w must be 2D, got shape={tuple(g_w.shape)}")

    n, D = g_w.shape

    q = int(q)
    if return_rank is None:
        return_rank = q
    return_rank = int(return_rank)

    q_work = min(
        max(1, q + int(oversample)),
        int(D),
        int(n),
    )
    return_rank = min(return_rank, q_work)

    device = g_w.device
    dtype = g_w.dtype

    # Random test matrix in input/noise dimension.
    Omega = torch.randn(
        int(D),
        int(q_work),
        device=device,
        dtype=dtype,
    )

    Y = g_w.t().matmul(g_w.matmul(Omega)) / float(n)
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional subspace / power iterations.
    for _ in range(int(niter)):
        Q, _ = torch.linalg.qr(Y, mode="reduced")
        Y = g_w.t().matmul(g_w.matmul(Q)) / float(n)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Orthonormal basis for approximate dominant range.
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    GQ = g_w.matmul(Q)
    B = GQ.t().matmul(GQ) / float(n)

    if float(jitter) > 0.0:
        B = B + float(jitter) * torch.eye(
            B.shape[0],
            device=device,
            dtype=dtype,
        )

    B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    B = 0.5 * (B + B.t())

    # Small q_work x q_work eigendecomposition on GPU.
    evals_small, V_small = torch.linalg.eigh(B)

    order = torch.argsort(evals_small, descending=True)
    order = order[:return_rank]

    evals = torch.clamp(evals_small[order], min=0.0)
    V_small = V_small[:, order]

    U = Q.matmul(V_small)
    U = U / (U.norm(dim=0, keepdim=True) + eps)

    return U, evals


@torch.no_grad()
def predict_proxy_objectives_np(X_norm_np, classifiers):
    """
    Predict normalized objective values using proxy models.

    Important:
        Return objectives, lower is better.
        Do NOT multiply by -1 for NDS.
    """
    preds = []

    for clf in classifiers:
        device = next(clf.parameters()).device
        dtype = next(clf.parameters()).dtype

        x_t = torch.as_tensor(X_norm_np, device=device, dtype=dtype)
        y_hat = clf(x_t).view(-1)
        preds.append(y_hat.detach().cpu().numpy())

    return np.stack(preds, axis=1)


def compute_hv_dict(task, task_name, X_denorm, Y_raw):
    Y_75 = get_quantile_solutions(Y_raw, 0.75)
    Y_50 = get_quantile_solutions(Y_raw, 0.50)

    nadir = task.nadir_point
    Y_norm = task.normalize_y(Y_raw, normalization_method="min-max")
    nad_norm = task.normalize_y(nadir, normalization_method="min-max")
    Y_75n = task.normalize_y(Y_75, normalization_method="min-max")
    Y_50n = task.normalize_y(Y_50, normalization_method="min-max")

    _, d_best = task.get_N_non_dominated_solutions(
        N=256,
        return_x=False,
        return_y=True,
    )
    d_best_n = task.normalize_y(d_best, normalization_method="min-max")

    hv_dbest = hv(nad_norm, d_best_n, task_name)
    hv_100 = hv(nad_norm, Y_norm, task_name)
    hv_75 = hv(nad_norm, Y_75n, task_name)
    hv_50 = hv(nad_norm, Y_50n, task_name)

    delta_100 = spread(Y_norm)
    delta_75 = spread(Y_75n)
    delta_50 = spread(Y_50n)

    hv_results = {
        "hypervolume/D(best)": hv_dbest,
        "hypervolume/100th": hv_100,
        "hypervolume/75th": hv_75,
        "hypervolume/50th": hv_50,
        "delta_spread/100th": delta_100,
        "delta_spread/75th": delta_75,
        "delta_spread/50th": delta_50,
        "eval/num_valid_returned": int(len(Y_raw)),
    }

    return hv_results, X_denorm, Y_raw


def _row_bad_mask(A):
    """
    Return a boolean mask of shape (N,), where True means the row/sample is bad.

    Works for:
        - (N, D)
        - (N, D1, D2)
        - (N,)
    """
    A = np.asarray(A)
    bad = np.logical_or(np.isinf(A), np.isnan(A))

    if A.ndim <= 1:
        return bad

    return np.any(bad, axis=tuple(range(1, A.ndim)))


def _ensure_y_matches_x_rows(Y, X, name="Y"):
    """
    Ensure objective array has shape (N, m), where N = len(X).

    If Y comes back as (m, N), transpose it.
    """
    Y = np.asarray(Y)
    N = len(X)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y.shape[0] != N and Y.ndim == 2 and Y.shape[1] == N:
        Y = Y.T

    if Y.shape[0] != N:
        raise RuntimeError(
            f"{name} has incompatible shape {Y.shape} for X with len={N}."
        )

    return Y


@torch.no_grad()
def evaluate_and_hv(
    task,
    task_name: str,
    X_norm_np: np.ndarray,
    num_solutions: int = 256,
    repair: bool = False,
    selection_mode: str = "none",  # "none", "proxy", "oracle"
    classifiers=None,
):
    selection_mode = str(selection_mode).lower()
    assert selection_mode in ["none", "proxy", "oracle"]

    X_norm_valid = np.asarray(X_norm_np)

    X_denorm = task.denormalize_x(X_norm_valid)

    if bool(repair) and (task.xl is not None) and (task.xu is not None):
        xl_raw = np.asarray(task.xl, dtype=X_denorm.dtype).reshape(1, -1)
        xu_raw = np.asarray(task.xu, dtype=X_denorm.dtype).reshape(1, -1)
        X_denorm = np.clip(X_denorm, xl_raw, xu_raw)

    if task.is_discrete:
        X_denorm = task.to_integers(X_denorm)

    bad_x = _row_bad_mask(X_denorm)
    valid_mask = np.logical_not(bad_x)

    X_norm_valid = X_norm_valid[valid_mask]
    X_denorm = X_denorm[valid_mask]

    if len(X_denorm) == 0:
        raise RuntimeError("No valid generated candidates after decoding/repair.")

    n_all = len(X_denorm)
    selected_idx = np.arange(n_all)

    if n_all > int(num_solutions):
        if selection_mode == "none":
            selected_idx = np.arange(n_all)

        elif selection_mode == "proxy":
            if classifiers is None:
                raise ValueError("selection_mode='proxy' requires classifiers.")

            # Score continuous candidates after final clipping, in the same
            # normalized coordinates used to train the proxies. NAS proxies
            # retain the generated continuous normalized logits, not integers.
            X_proxy = X_norm_valid
            if (
                bool(repair)
                and not task.is_discrete
                and task.xl is not None
                and task.xu is not None
            ):
                X_proxy = task.normalize_x(X_denorm)

            if not np.isfinite(X_proxy).all():
                raise FloatingPointError("Non-finite proxy selection inputs")

            Y_proxy = predict_proxy_objectives_np(
                X_proxy,
                classifiers,
            )

            if not np.isfinite(Y_proxy).all():
                raise FloatingPointError("Non-finite proxy selection predictions")

            fronts = NonDominatedSorting().do(Y_proxy)
            selected_idx = np.asarray(
                get_N_nondominated_indices(
                    Y=Y_proxy,
                    num_ret=int(num_solutions),
                    fronts=fronts,
                ),
                dtype=int,
            )

        elif selection_mode == "oracle":
            Y_raw_all = task.predict(X_denorm)
            Y_raw_all = _ensure_y_matches_x_rows(
                Y_raw_all,
                X_denorm,
                name="Y_raw_all",
            )

            bad_y = _row_bad_mask(Y_raw_all)
            good_y = np.logical_not(bad_y)

            X_norm_valid = X_norm_valid[good_y]
            X_denorm = X_denorm[good_y]
            Y_raw_all = Y_raw_all[good_y]

            if len(Y_raw_all) == 0:
                raise RuntimeError("No valid oracle-evaluated candidates.")

            fronts = NonDominatedSorting().do(Y_raw_all)
            selected_idx = np.asarray(
                get_N_nondominated_indices(
                    Y=Y_raw_all,
                    num_ret=min(int(num_solutions), len(Y_raw_all)),
                    fronts=fronts,
                ),
                dtype=int,
            )

            X_denorm = X_denorm[selected_idx]
            Y_raw = Y_raw_all[selected_idx]

            return compute_hv_dict(task, task_name, X_denorm, Y_raw)

    X_denorm = X_denorm[selected_idx]
    Y_raw = task.predict(X_denorm)
    Y_raw = _ensure_y_matches_x_rows(Y_raw, X_denorm, name="Y_raw")

    bad_y = _row_bad_mask(Y_raw)
    good_y = np.logical_not(bad_y)

    X_denorm = X_denorm[good_y]
    Y_raw = Y_raw[good_y]

    if len(Y_raw) == 0:
        raise RuntimeError("No valid true-evaluated candidates.")

    return compute_hv_dict(task, task_name, X_denorm, Y_raw)


@torch.no_grad()
def build_noise_label_dataset(
    flow,
    classifiers,
    n: int,
    batch_size: int = 256,
    xl=None,
    xu=None,
    repair: bool = False,
    t_threshold: float = 0.0,
    device=None,
):
    """
    Returns:
        Z: (n,D)
        Y: (n,m) or (n,m+1) if div_enable
    """
    if device is None:
        device = next(flow.parameters()).device

    D = flow.D
    m = len(classifiers)
    Z_all, Y_all = [], []

    for _ in range((n + batch_size - 1) // batch_size):
        b = min(batch_size, n - len(Z_all) * batch_size)
        if b <= 0:
            break

        z0 = torch.randn(b, D, device=device, dtype=torch.float32)
        x = flow.sample_from_noise(
            z0, xl=None, xu=None, repair=False, t_threshold=t_threshold
        )

        # Final x_1-space hard clipping only.
        if bool(repair) and (xl is not None) and (xu is not None):
            xl_t = torch.as_tensor(xl, device=x.device, dtype=x.dtype).view(1, -1)
            xu_t = torch.as_tensor(xu, device=x.device, dtype=x.dtype).view(1, -1)
            x = torch.minimum(torch.maximum(x, xl_t), xu_t)

        # ---- proxy normalized objectives -> scores ----
        ys = []
        for clf in classifiers:
            clf_dev = next(clf.parameters()).device
            clf_dtype = next(clf.parameters()).dtype
            x_in = x.to(device=clf_dev, dtype=clf_dtype)
            y_hat = clf(x_in)
            y_hat = y_hat.view(b) if y_hat.ndim > 1 else y_hat
            ys.append((-1.0) * y_hat.to(device=device, dtype=torch.float32))

        Yb = torch.stack(ys, dim=1)  # (b,m)

        Z_all.append(z0)
        Y_all.append(Yb)

    Z = torch.cat(Z_all, dim=0)[:n]
    Y = torch.cat(Y_all, dim=0)[:n]
    return Z, Y


@torch.no_grad()
def _proxy_scores_from_x(
    x: torch.Tensor,
    classifiers,
    out_device=None,
):
    """
    x: (B,D) in normalized x_1 space
    returns Y: (B,m), larger is better because score = - predicted objective
    """
    if out_device is None:
        out_device = x.device

    ys = []
    b = x.shape[0]

    for clf in classifiers:
        clf_dev = next(clf.parameters()).device
        clf_dtype = next(clf.parameters()).dtype
        x_in = x.to(device=clf_dev, dtype=clf_dtype)
        y_hat = clf(x_in)
        y_hat = y_hat.view(b) if y_hat.ndim > 1 else y_hat
        ys.append((-1.0) * y_hat.to(device=out_device, dtype=torch.float32))

    return torch.stack(ys, dim=1)  # (B,m)


@torch.no_grad()
def _build_dense_coeff_tensor(
    K: int,
    strengths_t: torch.Tensor,  # (S,)
    N: int,
    r: int,
    beta0: float = 0.50,
    decay: float = 0.50,
):
    """
    Build the dense coefficient tensor used for the full generation grid.
    Returns shape (K,S,N,r).
    """
    device = strengths_t.device
    dtype = strengths_t.dtype
    S = strengths_t.numel()

    a0 = strengths_t.view(1, S, 1).expand(K, S, N)  # (K,S,N)

    if r == 1:
        return a0.unsqueeze(-1)  # (K,S,N,1)

    betas = beta0 * (decay ** torch.arange(0, r - 1, device=device, dtype=dtype))
    aj = (strengths_t.view(1, S, 1, 1) * betas.view(1, 1, 1, r - 1)).expand(
        K, S, N, r - 1
    )

    return torch.cat([a0.unsqueeze(-1), aj], dim=-1)  # (K,S,N,r)


@torch.no_grad()
def _build_soft_full_rank_coeff_tensor(
    strengths_t: torch.Tensor,
    rho: torch.Tensor,
    N: int,
):
    """
    Full-rank soft spectral noise-steering coefficients.

    For each weight direction k and scalar strength s:

        x0 = z + sum_j s * rho_{k,j} * u_{k,j}

    Args:
        strengths_t: (S,)
        rho:         (K, D)
        N:           samples per weight/strength

    Returns:
        coeff:       (K, S, N, D)
    """
    K, D = rho.shape
    S = strengths_t.numel()

    coeff = strengths_t.view(1, S, 1, 1) * rho.view(K, 1, 1, D)

    coeff = coeff.expand(K, S, int(N), D).contiguous()

    return coeff


@torch.no_grad()
def _extract_adaptive_projector_bases(
    X_feat: torch.Tensor,
    Y: torch.Tensor,
    W_metric: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-12,
    jitter: float = 1e-6,
    low_rank_k: int = 32,
    low_rank_niter: int = 2,
):
    """
    Extract adaptive-hard RFM/AGOP projector bases.

    For each preference weight w:

        g_i(w) = G_i w                         # (D,)
        M_w    = (1/n) sum_i g_i(w) g_i(w)^T   # (D,D)

    For D <= 2000:
        use exact full SVD/eigendecomposition of M_w.

    For D > 2000:
        use torch.svd_lowrank on the symmetrized M_w. This still forms M_w,
        but avoids full D x D eigendecomposition.
    """
    device = X_feat.device
    dtype = X_feat.dtype

    n = int(G.shape[0])
    D = int(G.shape[1])
    K = int(W_metric.shape[0])

    use_low_rank = int(D) > 2000

    U_list = []
    ranks = []
    info_list = []

    for k in range(K):
        w = W_metric[k].to(device=device, dtype=dtype)
        w = w / (w.sum() + eps)

        # g_w: (n,D)
        g_w = torch.einsum("ndm,m->nd", G, w)
        g_w = torch.nan_to_num(g_w, nan=0.0, posinf=0.0, neginf=0.0)

        if use_low_rank:
            q_eff = min(
                max(1, int(low_rank_k)),
                int(D),
            )

            U_all, evals_all = randomized_psd_eig_from_gw_gpu(
                g_w=g_w,
                q=q_eff,
                niter=int(low_rank_niter),
                eps=eps,
                jitter=jitter,
                oversample=0,
                return_rank=q_eff,
            )

            U_all = U_all.to(device=device, dtype=dtype)
            evals_all = evals_all.to(device=device, dtype=dtype)

            spectrum_dim = int(evals_all.shape[0])
            basis_backend = "randomized_psd_eig_from_gw_gpu"

        else:
            # Scalarized RFM/AGOP matrix M_w.
            # Only form the dense D x D matrix for the exact full-SVD path.
            M_w = (g_w.t() @ g_w) / float(n)
            M_w = _sym(torch.nan_to_num(M_w, nan=0.0, posinf=0.0, neginf=0.0))

            U_all, evals_all = all_eigenvectors_svd(
                M_w,
                jitter=jitter,
                eps=eps,
                svd_on_cpu=True,
            )

            U_all = U_all.to(device=device, dtype=dtype)
            evals_all = evals_all.to(device=device, dtype=dtype)
            spectrum_dim = int(D)
            basis_backend = "exact_full_svd_Mw"

        if spectrum_dim <= 1:
            rank = 1
            min_gap_ratio = float("nan")
        else:
            denom = torch.clamp(evals_all[:-1], min=eps)
            ratios = evals_all[1:] / denom

            rank = int(torch.argmin(ratios).item()) + 1
            rank = max(1, min(rank, spectrum_dim))
            min_gap_ratio = float(ratios[rank - 1].detach().cpu())

        U = U_all[:, :rank].to(device=device, dtype=dtype)
        U = U / (U.norm(dim=0, keepdim=True) + eps)

        y_w = (Y @ w).view(-1)
        U, orient_corr = _orient_eigenvectors_by_label(
            U=U,
            X_feat=X_feat,
            y_w=y_w,
            eps=eps,
        )

        U_list.append(U)
        ranks.append(rank)

        info_list.append(
            {
                "adaptive_rank": int(rank),
                "rank": int(rank),
                "min_gap_ratio": min_gap_ratio,
                "eigval": evals_all.detach().cpu().numpy(),
                "orientation": "positive_corr_with_Yw",
                "orientation_corr": orient_corr,
                "basis_backend": basis_backend,
                "low_rank_active": bool(use_low_rank),
                "low_rank_k": int(low_rank_k),
                "low_rank_niter": int(low_rank_niter),
                "low_rank_effective_q": int(spectrum_dim),
                "w": w.detach().cpu().numpy(),
            }
        )

    r_max = max(ranks) if ranks else 1

    U_padded = torch.zeros(
        (len(U_list), D, r_max),
        device=device,
        dtype=dtype,
    )

    for k, U in enumerate(U_list):
        U_padded[k, :, : U.shape[1]] = U

    return U_padded, info_list


@torch.no_grad()
def _extract_adaptive_noise_steering_bases(
    X_feat: torch.Tensor,
    Y: torch.Tensor,
    W_metric: torch.Tensor,
    G: torch.Tensor,
    eps: float = 1e-12,
    jitter: float = 1e-6,
    low_rank_k: int = 32,
    low_rank_niter: int = 2,
):
    """
    Stage-1 adaptive-hard noise-steering bases for the selectable hard mode.

    This uses the same largest-spectral-gap rule as the Stage-2 projector:

        r = argmin_j lambda_{j+1} / lambda_j.

    The returned tensor is zero-padded to the maximum adaptive rank across
    preference directions.
    """
    U_padded, info_list = _extract_adaptive_projector_bases(
        X_feat=X_feat,
        Y=Y,
        W_metric=W_metric,
        G=G,
        eps=eps,
        jitter=jitter,
        low_rank_k=low_rank_k,
        low_rank_niter=low_rank_niter,
    )

    for info in info_list:
        info["stage"] = "stage1"
        info["mode"] = "adaptive_hard_noise_steering"

    return U_padded, info_list


@torch.no_grad()
def _extract_full_soft_spectral_projector_bases(
    X_feat: torch.Tensor,
    Y: torch.Tensor,
    W_metric: torch.Tensor,
    G: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-12,
    jitter: float = 1e-6,
    low_rank_k: int = 32,
    low_rank_niter: int = 2,
):
    """
    Soft spectral RFM/AGOP projector for guided flow.

    For each preference weight w:

        g_i(w) = G_i w                         # (D,)
        A_w    = (1/n) sum_i g_i(w) g_i(w)^T   # (D,D)
        A_w    = U diag(lambda) U^T

    For D <= 2000:
        use full-rank soft spectral RFM.

    For D > 2000:
        use low-rank soft spectral RFM by applying torch.svd_lowrank
        to the symmetrized A_w.
    """
    device = X_feat.device
    dtype = X_feat.dtype

    n = int(G.shape[0])
    D = int(G.shape[1])
    K = int(W_metric.shape[0])

    use_low_rank = int(D) > 2000

    U_list = []
    rho_list = []
    info_list = []

    for k in range(K):
        w = W_metric[k].to(device=device, dtype=dtype)
        w = w / (w.sum() + eps)

        # G: (n,D,m), w: (m,) -> g_w: (n,D)
        g_w = torch.einsum("ndm,m->nd", G, w)
        g_w = torch.nan_to_num(g_w, nan=0.0, posinf=0.0, neginf=0.0)

        if use_low_rank:
            q_eff = min(
                max(1, int(low_rank_k)),
                int(D),
            )

            U_all, lam = randomized_psd_eig_from_gw_gpu(
                g_w=g_w,
                q=q_eff,
                niter=int(low_rank_niter),
                eps=eps,
                jitter=jitter,
                oversample=0,
                return_rank=q_eff,
            )

            U_all = U_all.to(device=device, dtype=dtype)
            lam = lam.to(device=device, dtype=dtype)

            active_rank = int(lam.shape[0])
            basis_backend = "randomized_psd_eig_from_gw_gpu"
            full_rank = False

        else:
            # Preference-conditioned AGOP / RFM matrix.
            # Only form the dense D x D matrix for the exact full-SVD path.
            A_w = (g_w.t() @ g_w) / float(n)
            A_w = _sym(torch.nan_to_num(A_w, nan=0.0, posinf=0.0, neginf=0.0))

            # Exact full-rank path for normal-dimensional tasks.
            #
            # Keep jitter=0.0 here to preserve the original "actual spectrum"
            # behavior of the full-soft extractor.
            U_all, evals_all = all_eigenvectors_svd(
                A_w,
                jitter=0.0,
                eps=eps,
                svd_on_cpu=True,
            )

            U_all = U_all[:, :D].to(device=device, dtype=dtype)
            lam = torch.clamp(evals_all[:D].to(device=device, dtype=dtype), min=0.0)
            active_rank = int(D)
            basis_backend = "exact_full_svd_Mw"
            full_rank = True

        U_all = U_all[:, :active_rank]
        U_all = U_all / (U_all.norm(dim=0, keepdim=True) + eps)

        y_w = (Y @ w).view(-1)
        U_all, orient_corr = _orient_eigenvectors_by_label(
            U=U_all,
            X_feat=X_feat,
            y_w=y_w,
            eps=eps,
        )

        lam = torch.clamp(lam[:active_rank], min=0.0)
        lam1 = torch.clamp(lam[0], min=eps)

        rho = torch.pow(lam / lam1, float(gamma))
        rho = torch.nan_to_num(rho, nan=0.0, posinf=1.0, neginf=0.0)

        U_list.append(U_all)
        rho_list.append(rho)

        rho_sum = rho.sum()
        rho_eff_rank = (rho_sum * rho_sum) / (rho.pow(2).sum() + eps)

        info_list.append(
            {
                "adaptive_rank": int(active_rank),
                "rank": int(active_rank),
                "soft_rank": int(active_rank),
                "soft_full_rank": int(D) if full_rank else int(active_rank),
                "full_rank": bool(full_rank),
                "soft_low_rank": bool(use_low_rank),
                "basis_backend": basis_backend,
                "low_rank_active": bool(use_low_rank),
                "low_rank_k": int(low_rank_k),
                "low_rank_niter": int(low_rank_niter),
                "low_rank_effective_q": int(active_rank),
                "soft_gamma": float(gamma),
                "rho_min": float(rho.min().detach().cpu()),
                "rho_max": float(rho.max().detach().cpu()),
                "rho_mean": float(rho.mean().detach().cpu()),
                "rho_effective_rank": float(rho_eff_rank.detach().cpu()),
                "orientation": "positive_corr_with_Yw",
                "orientation_corr": orient_corr,
                "eigval": lam.detach().cpu().numpy(),
                "rho": rho.detach().cpu().numpy(),
                "w": w.detach().cpu().numpy(),
            }
        )

    r_max = max(U.shape[1] for U in U_list) if U_list else 1

    U_out = torch.zeros(
        (K, D, r_max),
        device=device,
        dtype=dtype,
    )

    rho_out = torch.zeros(
        (K, r_max),
        device=device,
        dtype=dtype,
    )

    for k, U in enumerate(U_list):
        r = U.shape[1]
        U_out[k, :, :r] = U
        rho_out[k, :r] = rho_list[k]

    return U_out, rho_out, info_list


def _rank_from_info(info_k, U_k) -> int:
    """
    Return active rank for saving/debugging.

    Adaptive-hard path:
        info_k["adaptive_rank"]

    Soft full-rank path:
        info_k["soft_full_rank"] == D

    U_k has shape (D, r).
    """
    if isinstance(info_k, dict):
        for key in ("adaptive_rank", "rank", "soft_rank", "soft_full_rank"):
            if key in info_k:
                return int(info_k[key])

    return int(U_k.shape[-1])


@torch.no_grad()
def _build_fixed_strength_coeff_tensor(
    steering_strength_t: torch.Tensor,
    batch_size: int,
    r: int,
    beta0: float = 0.50,
    decay: float = 0.50,
):
    """
    Build per-sample Stage-1 adaptive-hard coefficients using a fixed
    scalar steering strength.

    This is also the backward-compatible path when rho_stage1 is absent from
    APG's Stage-2 dataset builder.

    This keeps the current hard-rank coefficient decay:
        [s, 0.5s, 0.25s, ...]
    """
    device = steering_strength_t.device
    dtype = steering_strength_t.dtype

    a0 = steering_strength_t.expand(int(batch_size)).unsqueeze(-1)

    if int(r) == 1:
        return a0

    betas = beta0 * (
        decay
        ** torch.arange(
            0,
            int(r) - 1,
            device=device,
            dtype=dtype,
        )
    )

    aj = steering_strength_t * betas
    aj = aj.view(1, int(r) - 1).expand(int(batch_size), int(r) - 1)

    return torch.cat([a0, aj], dim=1)


@torch.no_grad()
def build_mid_state_label_dataset(
    flow,
    classifiers,
    U_stage1: torch.Tensor,
    steering_strength: float,
    n: int,
    t_mid: float = 0.6,
    batch_size: int = 256,
    xl=None,
    xu=None,
    repair: bool = False,
    t_threshold: float = 0.0,
    device=None,
    rho_stage1: torch.Tensor = None,
):
    """
    Build the stage-2 RFM dataset:
        z0 --(stage1 steering)--> z0'
        x_mid = Phi_{0->t_mid}(z0')
        x_fin = Phi_{t_mid->1}(x_mid)
        y = proxy score(x_fin)

    Soft-spectral APG supplies rho_stage1 and uses coefficients
    s * rho_stage1[k]. Adaptive-hard APG omits rho_stage1 and uses the
    historical geometric coefficient decay.

    Returns:
        X_mid: (n,D)
        Y:     (n,m)
    """
    if device is None:
        device = next(flow.parameters()).device

    D = flow.D
    K, D2, r1 = U_stage1.shape
    assert D2 == D

    if rho_stage1 is not None:
        if rho_stage1.ndim != 2 or tuple(rho_stage1.shape) != (K, r1):
            raise ValueError(
                "rho_stage1 must have shape (K, r1) matching U_stage1; "
                f"got rho_stage1={tuple(rho_stage1.shape)} and "
                f"U_stage1={tuple(U_stage1.shape)}."
            )

        rho_stage1 = rho_stage1.to(device=device, dtype=torch.float32)

    steering_strength_t = torch.tensor(
        float(steering_strength),
        device=device,
        dtype=torch.float32,
    )

    X_mid_all = []
    Y_all = []
    made = 0

    # Balanced coverage over Stage-1 dataset directions.
    #
    # This is preferable to purely random k_idx when K is large, because it
    # avoids accidentally under-sampling some W_metric_cache directions.
    perm = torch.randperm(K, device=device)
    perm_pos = 0

    while made < int(n):
        b = min(int(batch_size), int(n) - made)

        z0 = torch.randn(b, D, device=device, dtype=torch.float32)

        k_chunks = []
        remaining = b

        while remaining > 0:
            available = K - perm_pos
            take = min(remaining, available)

            k_chunks.append(perm[perm_pos : perm_pos + take])

            perm_pos += take
            remaining -= take

            if perm_pos >= K:
                perm = torch.randperm(K, device=device)
                perm_pos = 0

        k_idx = torch.cat(k_chunks, dim=0)

        if rho_stage1 is None:
            # Adaptive-hard path (also preserves historical standalone calls).
            coeff1 = _build_fixed_strength_coeff_tensor(
                steering_strength_t=steering_strength_t,
                batch_size=b,
                r=r1,
            )
        else:
            # Soft-spectral APG path: match the actual Stage-1 sampler exactly.
            # For preference k, coefficient j is s * rho[k, j].
            coeff1 = steering_strength_t * rho_stage1[k_idx]

        offsets1 = torch.einsum(
            "br,bdr->bd",
            coeff1,
            U_stage1[k_idx],
        )

        z0s = z0 + offsets1

        x_mid = flow.sample_segment(
            x_start=z0s,
            t_start=0.0,
            t_end=float(t_mid),
            xl=None,
            xu=None,
            repair=False,
            t_threshold=t_threshold,
        )

        x_fin = flow.sample_segment(
            x_start=x_mid,
            t_start=float(t_mid),
            t_end=1.0,
            xl=None,
            xu=None,
            repair=False,
            t_threshold=t_threshold,
        )

        # Final x_1-space hard clipping only.
        if bool(repair) and (xl is not None) and (xu is not None):
            xl_t = torch.as_tensor(xl, device=x_fin.device, dtype=x_fin.dtype).view(
                1, -1
            )
            xu_t = torch.as_tensor(xu, device=x_fin.device, dtype=x_fin.dtype).view(
                1, -1
            )
            x_fin = torch.minimum(torch.maximum(x_fin, xl_t), xu_t)

        Yb = _proxy_scores_from_x(x_fin, classifiers, out_device=device)

        X_mid_all.append(x_mid)
        Y_all.append(Yb)
        made += b

    X_mid = torch.cat(X_mid_all, dim=0)[:n]
    Y = torch.cat(Y_all, dim=0)[:n]

    return X_mid, Y


def _safe_cache_token(x) -> str:
    """
    Conservative filesystem-safe token.
    """
    s = str(x)
    for ch in ["/", "\\", " ", ":", ";", ","]:
        s = s.replace(ch, "_")
    return s


def _rfm_stage1_cache_path(
    cache_dir: str,
    cache_task_key: str,
    seed: int,
) -> str:
    """
    Stage-1 cache path.

    Example:
        dtlz1_0_RFM_stage1.pt

    The cache stores final shared-metric gradients G1, not eigenvectors.
    """
    os.makedirs(cache_dir, exist_ok=True)

    task_token = _safe_cache_token(cache_task_key)

    return os.path.join(
        cache_dir,
        f"{task_token}_{int(seed)}_RFM_stage1.pt",
    )


def _rfm_stage2_cache_path(
    cache_dir: str,
    cache_task_key: str,
    seed: int,
    stage1_steering_mode: str = "adaptive_hard",
) -> str:
    """
    Stage-2 cache path keyed by the Stage-1 midpoint distribution.

    APG couples its Stage-2 projector family to Stage-1. Hard/soft Stage-1
    modes produce different midpoint datasets and therefore use separate
    cache namespaces; no additional Stage-2 selector is needed in the key.
    """
    os.makedirs(cache_dir, exist_ok=True)

    task_token = _safe_cache_token(cache_task_key)

    stage1_steering_mode = str(stage1_steering_mode).lower()
    if stage1_steering_mode == "adaptive_hard":
        suffix = "RFM_stage2_fromStage1AdaptiveHard.pt"
    elif stage1_steering_mode == "soft_spectral":
        suffix = "RFM_stage2_fromStage1SoftSpectral.pt"
    else:
        raise ValueError(
            "stage1_steering_mode must be 'adaptive_hard' or "
            f"'soft_spectral'; got {stage1_steering_mode!r}."
        )

    return os.path.join(
        cache_dir,
        f"{task_token}_{int(seed)}_{suffix}",
    )


def _to_cpu_detached(obj):
    """
    Recursively move tensors to CPU before torch.save.
    This makes cache files device-independent.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: _to_cpu_detached(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_cpu_detached(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_to_cpu_detached(v) for v in obj)

    return obj


def _atomic_torch_save(obj, path: str):
    """
    Atomic save to avoid partially-written cache files.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp_path = path + f".tmp.{os.getpid()}"
    torch.save(_to_cpu_detached(obj), tmp_path)
    os.replace(tmp_path, path)


def _load_rfm_cache(path: str, device):
    return torch.load(path, map_location=device)


def _cuda_synchronize_if_needed(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _acquire_cache_lock(path: str, poll_seconds: float = 2.0):
    """
    Directory-based lock to prevent concurrent jobs from writing the same cache.
    """
    lock_dir = path + ".lock"

    while True:
        try:
            os.makedirs(lock_dir)
            return lock_dir
        except FileExistsError:
            print(f"[RFM Cache] Waiting for lock: {lock_dir}")
            sleep(float(poll_seconds))


def _cache_rebuild_hint(stage: int) -> str:
    if int(stage) == 2:
        return (
            "Use --retrain_rfm_stage2=True to rebuild only Stage-2, "
            "or --retrain_rfm=True to rebuild both stages. Hard and soft APG "
            "Stage-2 caches have distinct filenames in the shared cache "
            "directory."
        )

    return "Use --retrain_rfm=True to rebuild this cache."


def _check_cache_meta_or_raise(cache, expected_meta: dict, stage: int):
    """
    Strict cache metadata validation.

    Cache filenames intentionally do not include all RFM hyperparameters.
    This check prevents accidental reuse of stale RFM caches.
    """
    meta = cache.get("meta", {})

    mismatches = []
    for key, expected_value in expected_meta.items():
        cached_value = meta.get(key, None)
        if cached_value != expected_value:
            mismatches.append((key, cached_value, expected_value))

    if len(mismatches) > 0:
        msg = [
            f"[RFM Cache] Stage-{stage} cache metadata mismatch.",
            "The cache filename matched, but stored hyperparameters differ.",
            _cache_rebuild_hint(stage),
            "Mismatches:",
        ]

        for key, cached_value, expected_value in mismatches:
            msg.append(f"  - {key}: cached={cached_value}, current={expected_value}")

        raise RuntimeError("\n".join(msg))


def _check_cached_weight_tensor_or_raise(
    cached_w: torch.Tensor,
    current_w: torch.Tensor,
    stage: int,
):
    """
    For stage-2 caches, verify that the cached metric-weight tensor matches
    the current stage-1 metric-weight tensor.

    This matters when metric weights are random, for example Dirichlet.
    """
    cached_w = cached_w.detach().cpu()
    current_w = current_w.detach().cpu()

    if cached_w.shape != current_w.shape:
        raise RuntimeError(
            f"[RFM Cache] Stage-{stage} W_metric_cache shape mismatch: "
            f"cached={tuple(cached_w.shape)}, current={tuple(current_w.shape)}. "
            + _cache_rebuild_hint(stage)
        )

    if not torch.equal(cached_w, current_w):
        raise RuntimeError(
            f"[RFM Cache] Stage-{stage} W_metric_cache value mismatch. "
            "The metadata matched but the actual cached weight tensor differs. "
            + _cache_rebuild_hint(stage)
        )


def _check_cached_tensor_allclose_or_raise(
    cached_t: torch.Tensor,
    current_t: torch.Tensor,
    stage: int,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
):
    """
    Validate that a cached tensor matches the current tensor.

    Stage-2 data depends directly on its Stage-1 basis and spectral scales, so
    checking W_metric_cache alone is insufficient.
    """
    cached_t = cached_t.detach().cpu()
    current_t = current_t.detach().cpu()

    if cached_t.shape != current_t.shape:
        raise RuntimeError(
            f"[RFM Cache] Stage-{stage} {name} shape mismatch: "
            f"cached={tuple(cached_t.shape)}, current={tuple(current_t.shape)}. "
            + _cache_rebuild_hint(stage)
        )

    if not torch.allclose(
        cached_t,
        current_t,
        rtol=float(rtol),
        atol=float(atol),
    ):
        max_abs = torch.max(torch.abs(cached_t - current_t)).item()

        raise RuntimeError(
            f"[RFM Cache] Stage-{stage} {name} value mismatch. "
            f"max_abs_diff={max_abs:.6e}. "
            "The Stage-2 midpoint dataset depends on this tensor, so the cache "
            "is stale. " + _cache_rebuild_hint(stage)
        )


def _resolve_metric_weight_config(
    weight_mode: str,
    n_weights: int,
    metric_weight_mode: str = "same",
    metric_n_weights: int = 0,
):
    """
    Resolve the offline metric-recursion weight config.

    Semantics:
        metric_weight_mode == "same" and metric_n_weights <= 0:
            Use exactly the same tensor as sampling-time weights.
            This gives W_metric_cache == W_sample.

        metric_weight_mode == "same" and metric_n_weights > 0:
            Use the same weight-generation mode as sampling, but with
            metric_n_weights directions.
            This gives W_metric_cache != W_sample if metric_n_weights != n_weights.

        metric_weight_mode in {"dirichlet", "das-dennis", "per_obj"}:
            Use that explicit mode for the shared metric recursion.
    """
    metric_weight_mode = str(metric_weight_mode).lower()

    if metric_weight_mode == "same":
        resolved_mode = str(weight_mode)

        if metric_n_weights is None or int(metric_n_weights) <= 0:
            resolved_n_weights = int(n_weights)
            metric_same_as_sampling = True
        else:
            resolved_n_weights = int(metric_n_weights)
            metric_same_as_sampling = resolved_mode == str(
                weight_mode
            ) and resolved_n_weights == int(n_weights)

    else:
        resolved_mode = str(metric_weight_mode)

        if metric_n_weights is None or int(metric_n_weights) <= 0:
            resolved_n_weights = int(n_weights)
        else:
            resolved_n_weights = int(metric_n_weights)

        metric_same_as_sampling = resolved_mode == str(
            weight_mode
        ) and resolved_n_weights == int(n_weights)

    return resolved_mode, resolved_n_weights, metric_same_as_sampling


@torch.no_grad()
def _load_or_build_stage1_rfm_cache(
    flow,
    classifiers,
    cache_dir: str,
    cache_task_key: str,
    seed: int,
    retrain_rfm: bool,
    n_rfm_train: int,
    metric_weight_mode: str,
    metric_n_weights: int,
    dirichlet_alpha: float,
    dd_partitions: int,
    L: float,
    lam: float,
    T: int,
    xl=None,
    xu=None,
    repair: bool = False,
    t_threshold: float = 0.0,
    normalize_M_trace: bool = False,
    metric_jitter: float = 1e-6,
    eps: float = 1e-12,
):
    """
    Stage-1 offline RFM cache.

    Cached objects:
        Z
        Y
        W_metric_cache
        M_stage1
        G1

    G1 is the final shared-metric multi-output gradient tensor.
    It is the object needed to reconstruct per-weight AGOP/RFM subspaces.
    """
    device = next(flow.parameters()).device

    path = _rfm_stage1_cache_path(
        cache_dir=cache_dir,
        cache_task_key=cache_task_key,
        seed=seed,
    )

    expected_meta = {
        "cache_task_key": str(cache_task_key),
        "seed": int(seed),
        "stage": 1,
        "n_rfm_train": int(n_rfm_train),
        "metric_weight_mode": str(metric_weight_mode),
        "metric_n_weights": int(metric_n_weights),
        "dirichlet_alpha": float(dirichlet_alpha),
        "dd_partitions": int(dd_partitions),
        "L": float(L),
        "lam": float(lam),
        "T": int(T),
        "repair": bool(repair),
        "t_threshold": float(t_threshold),
        "normalize_M_trace": bool(normalize_M_trace),
        "metric_jitter": float(metric_jitter),
        "cache_version": 1,
    }

    load_seconds = 0.0
    compute_seconds = 0.0
    cache_hit = False

    if (not bool(retrain_rfm)) and os.path.exists(path):
        t0 = time()
        cache = _load_rfm_cache(path, device=device)
        load_seconds = time() - t0
        cache_hit = True

        _check_cache_meta_or_raise(
            cache,
            expected_meta,
            stage=1,
        )

        print(f"[RFM Cache] Loaded stage-1 cache: {path}")

        return cache, {
            "stage1_cache_path": path,
            "stage1_cache_hit": cache_hit,
            "stage1_cache_load_seconds": load_seconds,
            "stage1_cache_compute_seconds": compute_seconds,
        }

    lock_dir = _acquire_cache_lock(path)

    try:
        if (not bool(retrain_rfm)) and os.path.exists(path):
            t0 = time()
            cache = _load_rfm_cache(path, device=device)
            load_seconds = time() - t0
            cache_hit = True

            _check_cache_meta_or_raise(
                cache,
                expected_meta,
                stage=1,
            )

            print(f"[RFM Cache] Loaded stage-1 cache after wait: {path}")

            return cache, {
                "stage1_cache_path": path,
                "stage1_cache_hit": cache_hit,
                "stage1_cache_load_seconds": load_seconds,
                "stage1_cache_compute_seconds": compute_seconds,
            }

        print(f"[RFM Cache] Building stage-1 cache: {path}")
        t0 = time()

        Z, Y = build_noise_label_dataset(
            flow,
            classifiers,
            n=n_rfm_train,
            batch_size=min(n_rfm_train, 10000),
            xl=xl,
            xu=xu,
            repair=repair,
            t_threshold=t_threshold,
            device=device,
        )

        Z = Z.to(device=device, dtype=torch.float32)
        Y = Y.to(device=device, dtype=torch.float32)

        m = Y.shape[1]

        W_metric_cache = _make_weight_set(
            m=m,
            n_weights=metric_n_weights,
            weight_mode=metric_weight_mode,
            dirichlet_alpha=dirichlet_alpha,
            dd_partitions=dd_partitions,
            device=device,
            dtype=torch.float32,
        )

        M_stage1 = _shared_metric_rfm_recursion(
            Z=Z,
            Y=Y,
            W_metric=W_metric_cache,
            L=L,
            lam=lam,
            T=T,
            eps=eps,
            normalize_M_trace=normalize_M_trace,
            metric_jitter=metric_jitter,
        )

        grads1, _, _ = _compute_grads_multioutput(
            Z=Z,
            Y=Y,
            M=M_stage1,
            L=L,
            lam=lam,
            eps=eps,
            jitter=metric_jitter,
        )

        G1 = grads1.transpose(1, 2).contiguous()  # (n, D, m)

        compute_seconds = time() - t0

        cache = {
            "Z": Z,
            "Y": Y,
            "W_metric_cache": W_metric_cache,
            "M_stage1": M_stage1,
            "G1": G1,
            "meta": expected_meta,
        }

        _atomic_torch_save(cache, path)
        print(f"[RFM Cache] Saved stage-1 cache: {path}")

        return cache, {
            "stage1_cache_path": path,
            "stage1_cache_hit": cache_hit,
            "stage1_cache_load_seconds": load_seconds,
            "stage1_cache_compute_seconds": compute_seconds,
        }

    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


@torch.no_grad()
def _load_or_build_stage2_rfm_cache(
    flow,
    classifiers,
    U_stage1_for_dataset,
    rho_stage1_for_dataset,
    W_metric_cache,
    cache_dir: str,
    cache_task_key: str,
    seed: int,
    retrain_stage2: bool,
    n_rfm_train: int,
    steering_strength: float,
    soft_rfm_gamma: float,
    two_stage_t: float,
    metric_weight_mode: str,
    metric_n_weights: int,
    dirichlet_alpha: float,
    dd_partitions: int,
    L: float,
    lam: float,
    T: int,
    xl=None,
    xu=None,
    repair: bool = False,
    t_threshold: float = 0.0,
    normalize_M_trace: bool = False,
    metric_jitter: float = 1e-6,
    eps: float = 1e-12,
    stage1_steering_mode: str = "adaptive_hard",
):
    """
    Stage-2 offline RFM cache.

    Crucial behavior:
        Stage-2 cache is built from the selected Stage-1 midpoint
        distribution. Adaptive-hard uses U plus the historical geometric
        coefficient decay. Soft-spectral uses U and rho exactly as the online
        Stage-1 sampler does.

    Cached objects:
        X_mid_train
        Y_mid
        W_metric_cache
        U_stage1_for_dataset
        rho_stage1_for_dataset (soft-spectral Stage-1 only)
        M_stage2
        G2

    The cache does not store eigenvectors as the main object.
    G2 is the final shared-metric multi-output gradient tensor.
    """
    device = next(flow.parameters()).device

    stage1_steering_mode = str(stage1_steering_mode).lower()
    if stage1_steering_mode not in {"adaptive_hard", "soft_spectral"}:
        raise ValueError(
            "stage1_steering_mode must be 'adaptive_hard' or "
            f"'soft_spectral'; got {stage1_steering_mode!r}."
        )
    stage1_is_soft = stage1_steering_mode == "soft_spectral"

    if stage1_is_soft:
        if rho_stage1_for_dataset is None:
            raise ValueError(
                "rho_stage1_for_dataset is required for soft-spectral "
                "Stage-1 steering."
            )
        if rho_stage1_for_dataset.ndim != 2:
            raise ValueError(
                "rho_stage1_for_dataset must be a rank-2 tensor; "
                f"got shape={tuple(rho_stage1_for_dataset.shape)}."
            )

        expected_rho_shape = (
            int(U_stage1_for_dataset.shape[0]),
            int(U_stage1_for_dataset.shape[2]),
        )
        if tuple(rho_stage1_for_dataset.shape) != expected_rho_shape:
            raise ValueError(
                "rho_stage1_for_dataset must match the preference and "
                "basis-rank dimensions of U_stage1_for_dataset; "
                f"got rho={tuple(rho_stage1_for_dataset.shape)}, "
                f"U={tuple(U_stage1_for_dataset.shape)}."
            )
    elif rho_stage1_for_dataset is not None:
        raise ValueError(
            "rho_stage1_for_dataset must be None for adaptive-hard Stage-1 " "steering."
        )

    path = _rfm_stage2_cache_path(
        cache_dir=cache_dir,
        cache_task_key=cache_task_key,
        seed=seed,
        stage1_steering_mode=stage1_steering_mode,
    )

    steering_strength = float(steering_strength)

    expected_meta = {
        "cache_task_key": str(cache_task_key),
        "seed": int(seed),
        "stage": 2,
        "stage1_steering_mode": (
            "soft_spectral" if stage1_is_soft else "adaptive_hard_gap"
        ),
        "sampling_variant": "adaptive_projected_guidance",
        # Stage-2 data distribution.
        "stage2_dataset_weight_source": "W_metric_cache",
        "stage2_dataset_strength": float(steering_strength),
        "stage2_dataset_basis_shape": tuple(int(x) for x in U_stage1_for_dataset.shape),
        # Stage-2 RFM recursion.
        "n_rfm_train": int(n_rfm_train),
        "two_stage_t": float(two_stage_t),
        "metric_weight_mode": str(metric_weight_mode),
        "metric_n_weights": int(metric_n_weights),
        "dirichlet_alpha": float(dirichlet_alpha),
        "dd_partitions": int(dd_partitions),
        "L": float(L),
        "lam": float(lam),
        "T": int(T),
        "repair": bool(repair),
        "t_threshold": float(t_threshold),
        "normalize_M_trace": bool(normalize_M_trace),
        "metric_jitter": float(metric_jitter),
        "cache_version": 4 if stage1_is_soft else 3,
    }

    if stage1_is_soft:
        expected_meta.update(
            {
                "stage1_soft_rfm_gamma": float(soft_rfm_gamma),
                "stage2_dataset_spectral_scale_shape": tuple(
                    int(x) for x in rho_stage1_for_dataset.shape
                ),
                "stage2_dataset_coefficient_rule": "strength_times_rho",
            }
        )

    load_seconds = 0.0
    compute_seconds = 0.0
    cache_hit = False

    if (not bool(retrain_stage2)) and os.path.exists(path):
        t0 = time()
        cache = _load_rfm_cache(path, device=device)
        load_seconds = time() - t0
        cache_hit = True

        _check_cache_meta_or_raise(
            cache,
            expected_meta,
            stage=2,
        )

        _check_cached_weight_tensor_or_raise(
            cached_w=cache["W_metric_cache"],
            current_w=W_metric_cache,
            stage=2,
        )

        _check_cached_tensor_allclose_or_raise(
            cached_t=cache["U_stage1_for_dataset"],
            current_t=U_stage1_for_dataset,
            stage=2,
            name="U_stage1_for_dataset",
        )

        if stage1_is_soft:
            _check_cached_tensor_allclose_or_raise(
                cached_t=cache["rho_stage1_for_dataset"],
                current_t=rho_stage1_for_dataset,
                stage=2,
                name="rho_stage1_for_dataset",
            )

        print(f"[RFM Cache] Loaded stage-2 cache: {path}")

        return cache, {
            "stage2_cache_path": path,
            "stage2_cache_hit": cache_hit,
            "stage2_cache_load_seconds": load_seconds,
            "stage2_cache_compute_seconds": compute_seconds,
        }

    lock_dir = _acquire_cache_lock(path)

    try:
        if (not bool(retrain_stage2)) and os.path.exists(path):
            t0 = time()
            cache = _load_rfm_cache(path, device=device)
            load_seconds = time() - t0
            cache_hit = True

            _check_cache_meta_or_raise(
                cache,
                expected_meta,
                stage=2,
            )

            _check_cached_weight_tensor_or_raise(
                cached_w=cache["W_metric_cache"],
                current_w=W_metric_cache,
                stage=2,
            )

            _check_cached_tensor_allclose_or_raise(
                cached_t=cache["U_stage1_for_dataset"],
                current_t=U_stage1_for_dataset,
                stage=2,
                name="U_stage1_for_dataset",
            )

            if stage1_is_soft:
                _check_cached_tensor_allclose_or_raise(
                    cached_t=cache["rho_stage1_for_dataset"],
                    current_t=rho_stage1_for_dataset,
                    stage=2,
                    name="rho_stage1_for_dataset",
                )

            print(f"[RFM Cache] Loaded stage-2 cache after wait: {path}")

            return cache, {
                "stage2_cache_path": path,
                "stage2_cache_hit": cache_hit,
                "stage2_cache_load_seconds": load_seconds,
                "stage2_cache_compute_seconds": compute_seconds,
            }

        print(f"[RFM Cache] Building stage-2 cache: {path}")
        t0 = time()

        U_stage1_for_dataset = U_stage1_for_dataset.to(
            device=device,
            dtype=torch.float32,
        )

        if stage1_is_soft:
            rho_stage1_for_dataset = rho_stage1_for_dataset.to(
                device=device,
                dtype=torch.float32,
            )

        W_metric_cache = W_metric_cache.to(
            device=device,
            dtype=torch.float32,
        )

        X_mid_train, Y_mid = build_mid_state_label_dataset(
            flow=flow,
            classifiers=classifiers,
            U_stage1=U_stage1_for_dataset,
            rho_stage1=rho_stage1_for_dataset,
            steering_strength=float(steering_strength),
            n=n_rfm_train,
            t_mid=two_stage_t,
            batch_size=min(n_rfm_train, 10000),
            xl=xl,
            xu=xu,
            repair=repair,
            t_threshold=t_threshold,
            device=device,
        )

        X_mid_train = X_mid_train.to(device=device, dtype=torch.float32)
        Y_mid = Y_mid.to(device=device, dtype=torch.float32)

        M_stage2 = _shared_metric_rfm_recursion(
            Z=X_mid_train,
            Y=Y_mid,
            W_metric=W_metric_cache,
            L=L,
            lam=lam,
            T=T,
            eps=eps,
            normalize_M_trace=normalize_M_trace,
            metric_jitter=metric_jitter,
        )

        grads2, _, _ = _compute_grads_multioutput(
            Z=X_mid_train,
            Y=Y_mid,
            M=M_stage2,
            L=L,
            lam=lam,
            eps=eps,
            jitter=metric_jitter,
        )

        G2 = grads2.transpose(1, 2).contiguous()  # (n, D, m)

        compute_seconds = time() - t0

        cache = {
            "X_mid_train": X_mid_train,
            "Y_mid": Y_mid,
            "W_metric_cache": W_metric_cache,
            "U_stage1_for_dataset": U_stage1_for_dataset,
            "M_stage2": M_stage2,
            "G2": G2,
            "meta": expected_meta,
        }

        if stage1_is_soft:
            cache["rho_stage1_for_dataset"] = rho_stage1_for_dataset

        _atomic_torch_save(cache, path)
        print(f"[RFM Cache] Saved stage-2 cache: {path}")

        return cache, {
            "stage2_cache_path": path,
            "stage2_cache_hit": cache_hit,
            "stage2_cache_load_seconds": load_seconds,
            "stage2_cache_compute_seconds": compute_seconds,
        }

    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


@torch.no_grad()
def _compute_training_score_quantiles_from_task(
    task,
    classifiers,
    device,
    dtype=torch.float32,
    qlo: float = 0.01,
    qhi: float = 0.99,
    batch_size: int = 8192,
):
    """
    Compute empirical proxy-score support on offline training designs.

    Convention:
        classifier_i outputs minimization proxy objective f_hat_i.
        Online guidance uses maximization score s_i = -f_hat_i.

    Therefore this returns quantiles of:
        s_train_i = -classifier_i(X_train_norm)

    No new args are introduced.
    """
    X_np = task.x.copy()

    if task.is_discrete:
        X_np = task.to_logits(X_np)

    # Match the proxy-training input space used in train_proxies.
    # Your current experiments use normalized proxy inputs.
    X_np = task.normalize_x(X_np)

    X = torch.as_tensor(
        X_np,
        device=device,
        dtype=dtype,
    )

    score_chunks = []
    n = int(X.shape[0])

    for start in range(0, n, int(batch_size)):
        xb = X[start : start + int(batch_size)]

        cols = []
        for clf in classifiers:
            clf_device = next(clf.parameters()).device
            clf_dtype = next(clf.parameters()).dtype

            xb_clf = xb.to(device=clf_device, dtype=clf_dtype)

            pred = clf(xb_clf)
            pred = pred.view(pred.shape[0], -1)[:, 0]

            # Convert minimization objective prediction to maximization score.
            score = -pred.to(device=device, dtype=dtype)
            cols.append(score)

        score_chunks.append(torch.stack(cols, dim=1))

    scores_train = torch.cat(score_chunks, dim=0)  # (N,m)

    score_qlo = torch.quantile(scores_train, float(qlo), dim=0)
    score_qhi = torch.quantile(scores_train, float(qhi), dim=0)

    # Avoid degenerate zero-width intervals.
    width = score_qhi - score_qlo
    bad_width = width.abs() < 1e-8
    if bool(bad_width.any()):
        pad = torch.full_like(score_qlo, 1e-4)
        score_qlo = torch.where(bad_width, score_qlo - pad, score_qlo)
        score_qhi = torch.where(bad_width, score_qhi + pad, score_qhi)

    return score_qlo.detach(), score_qhi.detach()


RFM_STEERING_MODE_CHOICES = frozenset({"adaptive_hard", "soft_spectral"})

RFM_LEGACY_STAGE1_STEERING_MODE = {
    "adaptive_noise": "adaptive_hard",
    "adaptive_projected_guidance": "adaptive_hard",
    "data_adaptive_manifold_guidance": "soft_spectral",
    "crd_reduced": "soft_spectral",
    "cone_projected": "soft_spectral",
}


def resolve_rfm_steering_modes(
    sampling_variant: str,
    stage1_steering_mode=None,
    apg_stage2_projector_mode=None,
):
    """
    Resolve the selectable RFM steering family without changing legacy runs.

    Omitting ``stage1_steering_mode`` reproduces the repository's established
    behavior: adaptive-hard for adaptive noise/APG and soft-spectral for DAMG,
    CRD-Reduced, and Cone-Projected.  APG uses one coupled choice for both
    stages.  ``apg_stage2_projector_mode`` is retained only as a compatibility
    alias and may not request a mixed hard/soft APG configuration.
    """
    sampling_variant = str(sampling_variant).lower()
    if sampling_variant not in RFM_LEGACY_STAGE1_STEERING_MODE:
        raise ValueError(
            f"Unknown sampling_variant={sampling_variant!r}. "
            f"Expected one of {sorted(RFM_LEGACY_STAGE1_STEERING_MODE)}."
        )

    stage1_was_omitted = stage1_steering_mode is None
    if stage1_was_omitted:
        stage1_mode = RFM_LEGACY_STAGE1_STEERING_MODE[sampling_variant]
    else:
        stage1_mode = str(stage1_steering_mode).lower()
        if stage1_mode not in RFM_STEERING_MODE_CHOICES:
            raise ValueError(
                "stage1_steering_mode must be 'adaptive_hard' or "
                f"'soft_spectral'; got {stage1_mode!r}."
            )

    stage2_mode = None
    if sampling_variant == "adaptive_projected_guidance":
        if apg_stage2_projector_mode is None:
            stage2_mode = stage1_mode
        else:
            stage2_mode = str(apg_stage2_projector_mode).lower()
            if stage2_mode not in RFM_STEERING_MODE_CHOICES:
                raise ValueError(
                    "apg_stage2_projector_mode must be 'adaptive_hard' or "
                    f"'soft_spectral'; got {stage2_mode!r}."
                )
            if stage1_was_omitted:
                # Compatibility behavior: the deprecated Stage-2-only flag
                # now selects the single coupled APG family for both stages.
                stage1_mode = stage2_mode
            elif stage2_mode != stage1_mode:
                raise ValueError(
                    "adaptive_projected_guidance requires the same steering "
                    "family in Stage-1 and Stage-2; got "
                    f"Stage-1={stage1_mode!r}, Stage-2={stage2_mode!r}."
                )
    elif apg_stage2_projector_mode is not None:
        # Older wrappers passed this APG-only option to every variant. Keep
        # accepting a valid value outside APG, but deliberately ignore it.
        compatibility_mode = str(apg_stage2_projector_mode).lower()
        if compatibility_mode not in RFM_STEERING_MODE_CHOICES:
            raise ValueError(
                "apg_stage2_projector_mode must be 'adaptive_hard' or "
                f"'soft_spectral'; got {compatibility_mode!r}."
            )

    return stage1_mode, stage2_mode


@torch.no_grad()
def rfm_steered_flow_sample(
    flow,
    classifiers,
    task,
    task_name: str,
    cache_task_key: str = None,
    rfm_seed: int = 0,
    cache_dir: str = "experiments/results/rfm_cache_adaptive_variants_v3",
    retrain_rfm: bool = False,
    n_rfm_train: int = 8192,
    n_gen_per_obj: int = 512,
    strengths=(10.0,),
    weight_mode: str = "dirichlet",
    n_weights: int = 64,
    metric_weight_mode: str = "same",
    metric_n_weights: int = 0,
    dirichlet_alpha: float = 1.0,
    dd_partitions: int = 0,
    L: float = 10.0,
    lam: float = 1e-3,
    T: int = 5,
    num_solutions: int = 256,
    xl=None,
    xu=None,
    repair: bool = False,
    t_threshold: float = 0.0,
    normalize_M_trace: bool = False,
    metric_jitter: float = 1e-6,
    sampling_variant: str = "adaptive_projected_guidance",
    guidance_start_t: float = 0.7,
    guidance_alpha_max: float = 8.0,
    norm_match_beta: float = 0.25,
    soft_rfm_gamma: float = 1.0,
    tangent_start_t: float = 0.8,
    tangent_tau: float = 0.2,
    tangent_k_nn: int = 12,
    tangent_refresh_every: int = 1,
    knn_refresh_every: int = 15,
    tangent_verbose_log: bool = True,
    robust_gate_c: float = 0.1,
    low_rank_k: int = 32,
    low_rank_niter: int = 2,
    eps: float = 1e-12,
    retrain_rfm_stage2: bool = False,
    stage1_steering_mode=None,
    apg_stage2_projector_mode=None,
):
    """
    RFM-steered sampling with offline cache.

    Cache design:
        stage1 cache stores final shared-metric gradients G1.
        stage2 cache stores final shared-metric gradients G2 for APG and is
        namespaced by its Stage-1 midpoint distribution (adaptive-hard or
        soft-spectral). APG couples the Stage-1 and Stage-2 steering family.

        retrain_rfm rebuilds every required stage. retrain_rfm_stage2 rebuilds
        only APG Stage-2 and leaves a matching Stage-1 cache reusable.

    The cache does not store eigenvectors as the primary object.

    Online specialization:
        U_stage1 and U_stage2_projector are computed online from cached G1/G2
        and the sampling-time weights W_sample.

    Timing:
        time/rfm_steered_sampling_seconds measures only:
            - one-stage: ODE decoding from z0s to x1;
            - two-stage: ODE decoding to t_mid plus projected-guided ODE to x1.
            - data-adaptive / CRD variants: their complete online ODE sampler.

        It excludes U computation, eigenvector extraction, cache load, and cache construction.
        time/rfm_total_wall_seconds includes cache work, online specialization,
        decoding, and evaluation inside this function.
    """
    total_t0 = time()

    device = next(flow.parameters()).device
    D = flow.D
    m = len(classifiers)
    # Fix the returned candidate set using proxies before oracle evaluation.
    eval_selection_mode = "proxy"

    sampling_variant = str(sampling_variant).lower()

    single_stage_guided_variants = {
        "data_adaptive_manifold_guidance",
        "crd_reduced",
        "cone_projected",
    }

    if sampling_variant in single_stage_guided_variants and len(strengths) != 1:
        raise ValueError(
            f"{sampling_variant} expects exactly one scalar steering strength. "
            f"Got strengths={strengths}."
        )

    valid_variants = {
        "adaptive_noise",
        "adaptive_projected_guidance",
        "data_adaptive_manifold_guidance",
        "crd_reduced",
        "cone_projected",
    }

    if sampling_variant not in valid_variants:
        raise ValueError(
            f"Unknown sampling_variant={sampling_variant!r}. "
            f"Expected one of {sorted(valid_variants)}."
        )

    stage1_steering_mode, apg_stage2_projector_mode = resolve_rfm_steering_modes(
        sampling_variant=sampling_variant,
        stage1_steering_mode=stage1_steering_mode,
        apg_stage2_projector_mode=apg_stage2_projector_mode,
    )

    print(f"[RFM] Stage-1 steering mode: {stage1_steering_mode}")
    if sampling_variant == "adaptive_projected_guidance":
        print(
            "[RFM] APG coupled Stage-2 projector mode: " f"{apg_stage2_projector_mode}"
        )

    if cache_task_key is None:
        cache_task_key = task_name

    if sampling_variant != "adaptive_noise":
        assert (
            0.0 < float(guidance_start_t) < 1.0
        ), f"guidance_start_t must be in (0,1), got {guidance_start_t}"

    if sampling_variant == "data_adaptive_manifold_guidance":
        assert float(guidance_start_t) <= float(tangent_start_t) <= 1.0, (
            "For data_adaptive_manifold_guidance, tangent_start_t must satisfy "
            "guidance_start_t <= tangent_start_t <= 1. Got "
            f"{guidance_start_t} <= {tangent_start_t}."
        )
        assert (
            0.0 < float(tangent_tau) < 1.0
        ), f"tangent_tau must be in (0,1), got {tangent_tau}"
        tangent_k_nn = int(tangent_k_nn)
        tangent_refresh_every = max(1, int(tangent_refresh_every))
        knn_refresh_every = max(1, int(knn_refresh_every))
        assert tangent_k_nn >= 2, f"tangent_k_nn must be at least 2, got {tangent_k_nn}"

    assert (
        float(guidance_alpha_max) >= 0.0
    ), f"guidance_alpha_max must be nonnegative, got {guidance_alpha_max}"

    metric_weight_mode_resolved, metric_n_weights_resolved, metric_same_as_sampling = (
        _resolve_metric_weight_config(
            weight_mode=weight_mode,
            n_weights=n_weights,
            metric_weight_mode=metric_weight_mode,
            metric_n_weights=metric_n_weights,
        )
    )

    print("[RFM] Sampling weights: " f"mode={weight_mode}, n_weights={n_weights}")
    print(
        "[RFM] Metric-recursion weights: "
        f"mode={metric_weight_mode_resolved}, "
        f"n_weights={metric_n_weights_resolved}, "
        f"same_as_sampling={metric_same_as_sampling}"
    )

    # =========================================================
    # Stage 1 offline cache
    # =========================================================
    stage1_cache, stage1_cache_info = _load_or_build_stage1_rfm_cache(
        flow=flow,
        classifiers=classifiers,
        cache_dir=cache_dir,
        cache_task_key=cache_task_key,
        seed=rfm_seed,
        retrain_rfm=retrain_rfm,
        n_rfm_train=n_rfm_train,
        metric_weight_mode=metric_weight_mode_resolved,
        metric_n_weights=metric_n_weights_resolved,
        dirichlet_alpha=dirichlet_alpha,
        dd_partitions=dd_partitions,
        L=L,
        lam=lam,
        T=T,
        xl=xl,
        xu=xu,
        repair=repair,
        t_threshold=t_threshold,
        normalize_M_trace=normalize_M_trace,
        metric_jitter=metric_jitter,
        eps=eps,
    )

    Z = stage1_cache["Z"].to(device=device, dtype=torch.float32)
    Y = stage1_cache["Y"].to(device=device, dtype=torch.float32)
    M_stage1 = stage1_cache["M_stage1"].to(device=device, dtype=torch.float32)
    G1 = stage1_cache["G1"].to(device=device, dtype=torch.float32)
    W_metric_cache = stage1_cache["W_metric_cache"].to(
        device=device,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Sampling-time weights.
    #
    # If metric weights are "same", reuse the exact cached tensor.
    # This preserves old-RFM behavior where W_metric == W_sample.
    # ---------------------------------------------------------
    if metric_same_as_sampling:
        W_sample = W_metric_cache
    else:
        W_sample = _make_weight_set(
            m=m,
            n_weights=n_weights,
            weight_mode=weight_mode,
            dirichlet_alpha=dirichlet_alpha,
            dd_partitions=dd_partitions,
            device=device,
            dtype=torch.float32,
        )

    strengths_t = torch.tensor(
        list(strengths),
        device=device,
        dtype=torch.float32,
    )

    S = len(strengths)
    N = int(n_gen_per_obj)

    U_stage1_for_dataset = None
    rho_stage1_for_dataset = None
    info_stage1_for_dataset = None
    damg_online_U = None
    damg_online_rho = None
    damg_online_info = None

    if stage1_steering_mode == "adaptive_hard":
        if sampling_variant == "data_adaptive_manifold_guidance":
            damg_fork_devices = []
            if device.type == "cuda":
                device_index = device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                damg_fork_devices = [int(device_index)]

            with torch.random.fork_rng(
                devices=damg_fork_devices,
                enabled=True,
            ):
                damg_online_U, damg_online_rho, damg_online_info = (
                    _extract_full_soft_spectral_projector_bases(
                        X_feat=Z,
                        Y=Y,
                        W_metric=W_sample,
                        G=G1,
                        gamma=float(soft_rfm_gamma),
                        eps=eps,
                        jitter=metric_jitter,
                        low_rank_k=low_rank_k,
                        low_rank_niter=low_rank_niter,
                    )
                )

            damg_online_U = damg_online_U.to(
                device=device,
                dtype=torch.float32,
            )
            damg_online_rho = damg_online_rho.to(
                device=device,
                dtype=torch.float32,
            )

        U_stage1, info_stage1 = _extract_adaptive_noise_steering_bases(
            X_feat=Z,
            Y=Y,
            W_metric=W_sample,
            G=G1,
            eps=eps,
            jitter=metric_jitter,
            low_rank_k=low_rank_k,
            low_rank_niter=low_rank_niter,
        )

        U_stage1 = U_stage1.to(device=device, dtype=torch.float32)

        K = U_stage1.shape[0]
        r1 = U_stage1.shape[2]

        coeff1_dense = _build_dense_coeff_tensor(
            K,
            strengths_t,
            N,
            r1,
        )

        offsets1 = torch.einsum(
            "ksnr,kdr->ksnd",
            coeff1_dense,
            U_stage1,
        ).reshape(-1, D)

        rho_stage1 = None

        if sampling_variant == "adaptive_projected_guidance":
            if metric_same_as_sampling:
                U_stage1_for_dataset = U_stage1
                info_stage1_for_dataset = info_stage1
            else:
                U_stage1_for_dataset, info_stage1_for_dataset = (
                    _extract_adaptive_noise_steering_bases(
                        X_feat=Z,
                        Y=Y,
                        W_metric=W_metric_cache,
                        G=G1,
                        eps=eps,
                        jitter=metric_jitter,
                        low_rank_k=low_rank_k,
                        low_rank_niter=low_rank_niter,
                    )
                )

                U_stage1_for_dataset = U_stage1_for_dataset.to(
                    device=device,
                    dtype=torch.float32,
                )

    else:
        U_stage1, rho_stage1, info_stage1 = _extract_full_soft_spectral_projector_bases(
            X_feat=Z,
            Y=Y,
            W_metric=W_sample,
            G=G1,
            gamma=float(soft_rfm_gamma),
            eps=eps,
            jitter=metric_jitter,
            low_rank_k=low_rank_k,
            low_rank_niter=low_rank_niter,
        )

        U_stage1 = U_stage1.to(device=device, dtype=torch.float32)
        rho_stage1 = rho_stage1.to(device=device, dtype=torch.float32)

        K = U_stage1.shape[0]

        coeff1_dense = _build_soft_full_rank_coeff_tensor(
            strengths_t=strengths_t,
            rho=rho_stage1,
            N=N,
        )

        offsets1 = torch.einsum(
            "ksnr,kdr->ksnd",
            coeff1_dense,
            U_stage1,
        ).reshape(-1, D)

        if sampling_variant == "adaptive_projected_guidance":
            if metric_same_as_sampling:
                U_stage1_for_dataset = U_stage1
                rho_stage1_for_dataset = rho_stage1
                info_stage1_for_dataset = info_stage1
            else:
                (
                    U_stage1_for_dataset,
                    rho_stage1_for_dataset,
                    info_stage1_for_dataset,
                ) = _extract_full_soft_spectral_projector_bases(
                    X_feat=Z,
                    Y=Y,
                    W_metric=W_metric_cache,
                    G=G1,
                    gamma=float(soft_rfm_gamma),
                    eps=eps,
                    jitter=metric_jitter,
                    low_rank_k=low_rank_k,
                    low_rank_niter=low_rank_niter,
                )

                U_stage1_for_dataset = U_stage1_for_dataset.to(
                    device=device,
                    dtype=torch.float32,
                )
                rho_stage1_for_dataset = rho_stage1_for_dataset.to(
                    device=device,
                    dtype=torch.float32,
                )
        elif sampling_variant == "data_adaptive_manifold_guidance":
            # Reuse the exact Stage-1 soft basis in the established DAMG
            # online filter; no duplicate eigendecomposition is introduced.
            damg_online_U = U_stage1
            damg_online_rho = rho_stage1
            damg_online_info = info_stage1

    k_ids = (
        torch.arange(
            K,
            device=device,
        )
        .view(K, 1, 1)
        .expand(K, S, N)
        .reshape(-1)
    )

    z0 = torch.randn(K * S * N, D, device=device, dtype=torch.float32)
    z0s = z0 + offsets1

    # =========================================================
    # One-stage path
    #
    # Timer measures only ODE decoding.
    # =========================================================
    if sampling_variant == "adaptive_noise":
        _cuda_synchronize_if_needed(device)
        decode_t0 = time()

        X_norm = flow.sample_from_noise(
            z0s,
            xl=None,
            xu=None,
            repair=False,
            t_threshold=t_threshold,
        )

        _cuda_synchronize_if_needed(device)
        steered_sampling_seconds = time() - decode_t0

        eval_t0 = time()

        X_norm_np = X_norm.detach().cpu().numpy()

        hv_results, X_denorm, Y_raw = evaluate_and_hv(
            task,
            task_name,
            X_norm_np,
            num_solutions=num_solutions,
            repair=repair,
            selection_mode=eval_selection_mode,
            classifiers=classifiers,
        )

        eval_seconds = time() - eval_t0

        hv_results["time/rfm_stage1_cache_load_seconds"] = stage1_cache_info[
            "stage1_cache_load_seconds"
        ]
        hv_results["time/rfm_stage1_cache_compute_seconds"] = stage1_cache_info[
            "stage1_cache_compute_seconds"
        ]
        hv_results["time/rfm_offline_compute_seconds"] = stage1_cache_info[
            "stage1_cache_compute_seconds"
        ]
        hv_results["time/rfm_cache_load_seconds"] = stage1_cache_info[
            "stage1_cache_load_seconds"
        ]

        # Important: excludes U/eigenvector computation.
        hv_results["time/rfm_steered_sampling_seconds"] = steered_sampling_seconds
        hv_results["time/rfm_ode_decoding_seconds"] = steered_sampling_seconds
        hv_results["time/rfm_evaluation_seconds"] = eval_seconds
        hv_results["time/rfm_total_wall_seconds"] = time() - total_t0

        hv_results["rfm/cache_stage1_hit"] = bool(stage1_cache_info["stage1_cache_hit"])
        hv_results["rfm/cache_stage1_path"] = stage1_cache_info["stage1_cache_path"]
        hv_results["rfm/cache_dir"] = str(cache_dir)

        hv_results["rfm/sampling_variant"] = "adaptive_noise"
        hv_results["rfm/stage1_steering_family"] = stage1_steering_mode
        hv_results["rfm/stage1_steering_mode"] = (
            "adaptive_hard_gap"
            if stage1_steering_mode == "adaptive_hard"
            else ("soft_low_rank_svd_Mw" if int(D) > 2000 else "soft_full_rank")
        )
        hv_results["rfm/stage2_used"] = False
        hv_results["rfm/guidance_used"] = False
        hv_results["rfm/manual_subspace_k_used"] = False
        hv_results["rfm/norm_match_beta"] = 0.0
        hv_results["rfm/soft_rfm_gamma"] = (
            float(soft_rfm_gamma) if stage1_steering_mode == "soft_spectral" else 0.0
        )
        hv_results["rfm/soft_rfm_rank"] = (
            int(U_stage1.shape[-1]) if stage1_steering_mode == "soft_spectral" else 0
        )

        hv_results["rfm/metric_weight_mode"] = str(metric_weight_mode_resolved)
        hv_results["rfm/metric_n_weights"] = int(metric_n_weights_resolved)
        hv_results["rfm/metric_same_as_sampling"] = bool(metric_same_as_sampling)
        hv_results["rfm/sampling_weight_mode"] = str(weight_mode)
        hv_results["rfm/sampling_n_weights"] = int(n_weights)

        hv_results["rfm/low_rank_active"] = bool(int(D) > 2000)
        hv_results["rfm/low_rank_backend"] = (
            "svd_lowrank_Mw" if int(D) > 2000 else "exact_full_svd_Mw"
        )
        hv_results["rfm/low_rank_k"] = int(low_rank_k)
        hv_results["rfm/low_rank_niter"] = int(low_rank_niter)
        hv_results["rfm/stage1_basis_rank_max"] = int(U_stage1.shape[-1])
        hv_results["eval/selection_mode"] = str(eval_selection_mode)
        hv_results["eval/num_solutions"] = int(num_solutions)

        return {
            "hv_results": hv_results,
            "X_denorm": X_denorm,
            "Y_raw": Y_raw,
            "U_list": [
                U_stage1[
                    k,
                    :,
                    : _rank_from_info(
                        info_stage1[k],
                        U_stage1[k],
                    ),
                ]
                .detach()
                .cpu()
                .numpy()
                for k in range(U_stage1.shape[0])
            ],
            "rfm_info": info_stage1,
            "rho_stage1": (
                rho_stage1.detach().cpu().numpy() if rho_stage1 is not None else None
            ),
            "M_final": None if int(D) > 2000 else M_stage1.detach().cpu().numpy(),
            "W": W_sample.detach().cpu().numpy(),
        }

    if sampling_variant in single_stage_guided_variants:
        decode_weights = W_sample[k_ids].to(device=device, dtype=torch.float32)
        damg_tangent_stats = None

        if sampling_variant == "data_adaptive_manifold_guidance":
            decode_basis = damg_online_U[k_ids].to(
                device=device,
                dtype=torch.float32,
            )
            decode_scales = damg_online_rho[k_ids].to(
                device=device,
                dtype=torch.float32,
            )

        _cuda_synchronize_if_needed(device)
        decode_t0 = time()

        if sampling_variant == "data_adaptive_manifold_guidance":
            X_norm, damg_tangent_stats = (
                flow.sample_rfm_data_adaptive_manifold_guided_knn(
                    x_start=z0s,
                    guidance_start_t=float(guidance_start_t),
                    classifiers=classifiers,
                    weights=decode_weights,
                    rfm_basis=decode_basis,
                    rfm_scales=decode_scales,
                    alpha_clip=float(guidance_alpha_max),
                    norm_match_beta=float(norm_match_beta),
                    tangent_start_t=float(tangent_start_t),
                    tangent_tau=float(tangent_tau),
                    tangent_k_nn=int(tangent_k_nn),
                    tangent_refresh_every=int(tangent_refresh_every),
                    knn_refresh_every=int(knn_refresh_every),
                    xl=None,
                    xu=None,
                    repair=False,
                    t_threshold=t_threshold,
                    eps=eps,
                    log_tag=f"{task_name} seed{rfm_seed}",
                    verbose_tangent_log=bool(tangent_verbose_log),
                    return_stats=True,
                )
            )
        elif sampling_variant == "crd_reduced":
            X_norm = flow.sample_crd_reduced_guided(
                x_start=z0s,
                guidance_start_t=float(guidance_start_t),
                classifiers=classifiers,
                weights=decode_weights,
                robust_gate_c=float(robust_gate_c),
                norm_match_beta=float(norm_match_beta),
                xl=None,
                xu=None,
                repair=False,
                t_threshold=t_threshold,
                eps=eps,
            )
        else:
            assert sampling_variant == "cone_projected"
            X_norm = flow.sample_cone_projected_guided(
                x_start=z0s,
                guidance_start_t=float(guidance_start_t),
                classifiers=classifiers,
                weights=decode_weights,
                norm_match_beta=float(norm_match_beta),
                xl=None,
                xu=None,
                repair=False,
                t_threshold=t_threshold,
                eps=eps,
            )

        _cuda_synchronize_if_needed(device)
        steered_sampling_seconds = time() - decode_t0

        eval_t0 = time()

        X_norm_np = X_norm.detach().cpu().numpy()

        hv_results, X_denorm, Y_raw = evaluate_and_hv(
            task,
            task_name,
            X_norm_np,
            num_solutions=num_solutions,
            repair=repair,
            selection_mode=eval_selection_mode,
            classifiers=classifiers,
        )

        eval_seconds = time() - eval_t0

        hv_results["time/rfm_stage1_cache_load_seconds"] = stage1_cache_info[
            "stage1_cache_load_seconds"
        ]
        hv_results["time/rfm_stage1_cache_compute_seconds"] = stage1_cache_info[
            "stage1_cache_compute_seconds"
        ]
        hv_results["time/rfm_offline_compute_seconds"] = stage1_cache_info[
            "stage1_cache_compute_seconds"
        ]
        hv_results["time/rfm_cache_load_seconds"] = stage1_cache_info[
            "stage1_cache_load_seconds"
        ]
        hv_results["time/rfm_steered_sampling_seconds"] = steered_sampling_seconds
        hv_results[f"time/rfm_{sampling_variant.replace('-', '_')}_seconds"] = (
            steered_sampling_seconds
        )
        hv_results["time/rfm_evaluation_seconds"] = eval_seconds
        hv_results["time/rfm_total_wall_seconds"] = time() - total_t0

        is_data_adaptive = sampling_variant == "data_adaptive_manifold_guidance"
        is_crd_reduced = sampling_variant == "crd_reduced"
        is_cone_projected = sampling_variant == "cone_projected"

        hv_results["rfm/sampling_variant"] = sampling_variant
        hv_results["rfm/stage1_steering_family"] = stage1_steering_mode
        hv_results["rfm/stage1_steering_mode"] = (
            "adaptive_hard_gap"
            if stage1_steering_mode == "adaptive_hard"
            else ("soft_low_rank_svd_Mw" if int(D) > 2000 else "soft_full_rank")
        )
        stage1_soft_rank = (
            int(U_stage1.shape[-1]) if stage1_steering_mode == "soft_spectral" else 0
        )
        damg_online_soft_rank = int(damg_online_U.shape[-1]) if is_data_adaptive else 0
        # Historical keys continue to describe the active soft-RFM component.
        # For hard-Stage-1 DAMG this is its unchanged online gradient filter.
        hv_results["rfm/soft_rfm_rank"] = (
            damg_online_soft_rank if is_data_adaptive else stage1_soft_rank
        )
        hv_results["rfm/stage2_used"] = False
        hv_results["rfm/guidance_gradient_location"] = "predicted_endpoint_xhat1"
        hv_results["rfm/soft_rfm_gamma"] = (
            float(soft_rfm_gamma)
            if stage1_steering_mode == "soft_spectral" or is_data_adaptive
            else 0.0
        )
        hv_results["rfm/stage1_soft_rfm_rank"] = stage1_soft_rank
        hv_results["rfm/stage1_soft_rfm_gamma"] = (
            float(soft_rfm_gamma) if stage1_steering_mode == "soft_spectral" else 0.0
        )
        hv_results["rfm/damg_online_rfm_filter_family"] = (
            "soft_spectral" if is_data_adaptive else "none"
        )
        hv_results["rfm/damg_online_soft_rfm_rank"] = damg_online_soft_rank
        hv_results["rfm/damg_online_soft_rfm_gamma"] = (
            float(soft_rfm_gamma) if is_data_adaptive else 0.0
        )
        hv_results["rfm/norm_match_beta"] = float(norm_match_beta)
        hv_results["rfm/guidance_start_t"] = float(guidance_start_t)
        hv_results["rfm/guidance_alpha_clip"] = (
            float(guidance_alpha_max) if is_data_adaptive else None
        )
        hv_results["rfm/online_soft_rfm_filter_used"] = bool(is_data_adaptive)
        hv_results["rfm/tangent_projection_used"] = bool(is_data_adaptive)
        hv_results["rfm/tangent_estimator"] = (
            "state_space_knn_local_gram_pca_cached_logged"
            if is_data_adaptive
            else "none"
        )
        hv_results["rfm/tangent_start_t"] = (
            float(tangent_start_t) if is_data_adaptive else 0.0
        )
        hv_results["rfm/tangent_tau"] = float(tangent_tau) if is_data_adaptive else 0.0
        hv_results["rfm/tangent_k_nn"] = int(tangent_k_nn) if is_data_adaptive else 0
        hv_results["rfm/tangent_refresh_every"] = (
            int(tangent_refresh_every) if is_data_adaptive else 0
        )
        hv_results["rfm/knn_refresh_every"] = (
            int(knn_refresh_every) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_requires_shared_preference"] = (
            False if is_data_adaptive else None
        )
        hv_results["rfm/damg_reference_variant"] = (
            "fixed_localspace_fast_r1_logged" if is_data_adaptive else "none"
        )
        hv_results["rfm/tangent_logging_enabled"] = (
            bool(tangent_verbose_log) if is_data_adaptive else False
        )
        hv_results["rfm/tangent_proj_steps"] = (
            int(damg_tangent_stats["proj_steps"]) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_noop_steps"] = (
            int(damg_tangent_stats["noop_steps"]) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_guided_proj_window"] = (
            int(damg_tangent_stats["guided_proj_window"]) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_noop_frac"] = (
            float(damg_tangent_stats["noop_frac"]) if is_data_adaptive else None
        )
        hv_results["rfm/tangent_refreshes"] = (
            int(damg_tangent_stats["refreshes"]) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_certified_refreshes"] = (
            int(damg_tangent_stats["certified_refreshes"]) if is_data_adaptive else 0
        )
        hv_results["rfm/tangent_rank_trace"] = (
            damg_tangent_stats["rank_trace"] if is_data_adaptive else []
        )
        hv_results["rfm/tangent_noop_reasons"] = (
            damg_tangent_stats["noop_reasons"] if is_data_adaptive else {}
        )
        hv_results["rfm/tangent_logged_config"] = (
            damg_tangent_stats["config"] if is_data_adaptive else {}
        )
        # Preserve this historical key for downstream schema compatibility.
        # It is intentionally null because local state-space neighborhoods
        # support batches that mix preference directions.
        hv_results["rfm/batch_tangent_shared_preference"] = None
        hv_results["rfm/crd_variant"] = (
            "reduced" if is_crd_reduced else "cone" if is_cone_projected else "none"
        )
        hv_results["rfm/normalized_objective_jacobian_used"] = bool(
            is_crd_reduced or is_cone_projected
        )
        hv_results["rfm/robust_direction_used"] = bool(
            is_crd_reduced or is_cone_projected
        )
        hv_results["rfm/crd_gate_c"] = float(robust_gate_c) if is_crd_reduced else 0.0
        hv_results["rfm/manual_subspace_k_used"] = False

        hv_results["rfm/cache_stage1_hit"] = bool(stage1_cache_info["stage1_cache_hit"])
        hv_results["rfm/cache_stage1_path"] = stage1_cache_info["stage1_cache_path"]
        hv_results["rfm/cache_dir"] = str(cache_dir)

        hv_results["rfm/metric_weight_mode"] = str(metric_weight_mode_resolved)
        hv_results["rfm/metric_n_weights"] = int(metric_n_weights_resolved)
        hv_results["rfm/metric_same_as_sampling"] = bool(metric_same_as_sampling)
        hv_results["rfm/sampling_weight_mode"] = str(weight_mode)
        hv_results["rfm/sampling_n_weights"] = int(n_weights)

        hv_results["rfm/low_rank_active"] = bool(int(D) > 2000)
        hv_results["rfm/low_rank_backend"] = (
            "svd_lowrank_Mw" if int(D) > 2000 else "exact_full_svd_Mw"
        )
        hv_results["rfm/low_rank_k"] = int(low_rank_k)
        hv_results["rfm/low_rank_niter"] = int(low_rank_niter)
        hv_results["rfm/stage1_basis_rank_max"] = int(U_stage1.shape[-1])
        hv_results["eval/selection_mode"] = str(eval_selection_mode)
        hv_results["eval/num_solutions"] = int(num_solutions)

        return {
            "hv_results": hv_results,
            "X_denorm": X_denorm,
            "Y_raw": Y_raw,
            "U_list": [
                U_stage1[
                    k,
                    :,
                    : _rank_from_info(
                        info_stage1[k],
                        U_stage1[k],
                    ),
                ]
                .detach()
                .cpu()
                .numpy()
                for k in range(U_stage1.shape[0])
            ],
            "rfm_info": info_stage1,
            "rho_stage1": (
                rho_stage1.detach().cpu().numpy() if rho_stage1 is not None else None
            ),
            "damg_online_rfm_info": damg_online_info,
            "M_final": None if int(D) > 2000 else M_stage1.detach().cpu().numpy(),
            "W": W_sample.detach().cpu().numpy(),
        }

    # =========================================================
    # Adaptive projected-guidance path.
    #
    # Stage-2 cache follows the selected Stage-1 midpoint distribution, and
    # the online Stage-2 projector is forced to use the same steering family.
    # There is no manually specified top-1/top-3 rank in either mode.
    # =========================================================
    assert sampling_variant == "adaptive_projected_guidance"

    if len(strengths) != 1:
        raise ValueError(
            "The cleaned-up Stage-2 RFM dataset logic expects exactly one "
            f"Stage-1 steering strength. Got strengths={strengths}."
        )

    stage2_dataset_strength = float(strengths[0])

    # Soft APG is new, so isolate its random Stage-2 cache-construction draws:
    # cache-build and cache-hit runs then enter the randomized online Stage-2
    # eigensolver with the same RNG state. Keep this disabled for the legacy
    # adaptive-hard path so its cache-miss RNG semantics remain unchanged.
    isolate_stage2_cache_rng = stage1_steering_mode == "soft_spectral"
    fork_devices = []
    if isolate_stage2_cache_rng and device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        fork_devices = [int(device_index)]

    with torch.random.fork_rng(
        devices=fork_devices,
        enabled=isolate_stage2_cache_rng,
    ):
        stage2_cache, stage2_cache_info = _load_or_build_stage2_rfm_cache(
            flow=flow,
            classifiers=classifiers,
            U_stage1_for_dataset=U_stage1_for_dataset,
            rho_stage1_for_dataset=rho_stage1_for_dataset,
            W_metric_cache=W_metric_cache,
            cache_dir=cache_dir,
            cache_task_key=cache_task_key,
            seed=rfm_seed,
            retrain_stage2=(bool(retrain_rfm) or bool(retrain_rfm_stage2)),
            n_rfm_train=n_rfm_train,
            steering_strength=stage2_dataset_strength,
            soft_rfm_gamma=float(soft_rfm_gamma),
            two_stage_t=guidance_start_t,
            metric_weight_mode=metric_weight_mode_resolved,
            metric_n_weights=metric_n_weights_resolved,
            dirichlet_alpha=dirichlet_alpha,
            dd_partitions=dd_partitions,
            L=L,
            lam=lam,
            T=T,
            xl=xl,
            xu=xu,
            repair=repair,
            t_threshold=t_threshold,
            normalize_M_trace=normalize_M_trace,
            metric_jitter=metric_jitter,
            eps=eps,
            stage1_steering_mode=stage1_steering_mode,
        )

    X_mid_train = stage2_cache["X_mid_train"].to(device=device, dtype=torch.float32)
    Y_mid = stage2_cache["Y_mid"].to(device=device, dtype=torch.float32)
    M_stage2 = stage2_cache["M_stage2"].to(device=device, dtype=torch.float32)
    G2 = stage2_cache["G2"].to(device=device, dtype=torch.float32)

    # =========================================================
    # Online stage-2 specialization from cached G2
    #
    # This is intentionally NOT included in rfm_steered_sampling_seconds.
    # =========================================================
    if apg_stage2_projector_mode == "adaptive_hard":
        U_stage2_projector, info_stage2 = _extract_adaptive_projector_bases(
            X_feat=X_mid_train,
            Y=Y_mid,
            W_metric=W_sample,
            G=G2,
            eps=eps,
            jitter=metric_jitter,
            low_rank_k=low_rank_k,
            low_rank_niter=low_rank_niter,
        )
        rho_stage2_projector = None
    else:
        (
            U_stage2_projector,
            rho_stage2_projector,
            info_stage2,
        ) = _extract_full_soft_spectral_projector_bases(
            X_feat=X_mid_train,
            Y=Y_mid,
            W_metric=W_sample,
            G=G2,
            gamma=float(soft_rfm_gamma),
            eps=eps,
            jitter=metric_jitter,
            low_rank_k=low_rank_k,
            low_rank_niter=low_rank_niter,
        )

        U_stage2_projector = U_stage2_projector.to(
            device=device,
            dtype=torch.float32,
        )
        rho_stage2_projector = rho_stage2_projector.to(
            device=device,
            dtype=torch.float32,
        )

    decode_weights = W_sample[k_ids].to(device=device, dtype=torch.float32)

    decode_projectors = U_stage2_projector[k_ids].to(
        device=device,
        dtype=torch.float32,
    )
    decode_projector_scales = (
        rho_stage2_projector[k_ids].to(device=device, dtype=torch.float32)
        if rho_stage2_projector is not None
        else None
    )

    _cuda_synchronize_if_needed(device)
    decode_t0 = time()

    X_mid = flow.sample_segment(
        x_start=z0s,
        t_start=0.0,
        t_end=float(guidance_start_t),
        xl=None,
        xu=None,
        repair=False,
        t_threshold=t_threshold,
    )

    X_norm = flow.sample_segment_rfm_projected_guided(
        x_start=X_mid,
        t_start=float(guidance_start_t),
        t_end=1.0,
        classifiers=classifiers,
        weights=decode_weights,
        projector_basis=decode_projectors,
        projector_scales=decode_projector_scales,
        alpha_max=float(guidance_alpha_max),
        norm_match_enable=False,
        norm_match_beta=0.0,
        xl=None,
        xu=None,
        repair=False,
        t_threshold=t_threshold,
        eps=eps,
    )

    _cuda_synchronize_if_needed(device)
    steered_sampling_seconds = time() - decode_t0

    eval_t0 = time()

    X_norm_np = X_norm.detach().cpu().numpy()

    hv_results, X_denorm, Y_raw = evaluate_and_hv(
        task,
        task_name,
        X_norm_np,
        num_solutions=num_solutions,
        repair=repair,
        selection_mode=eval_selection_mode,
        classifiers=classifiers,
    )

    eval_seconds = time() - eval_t0

    hv_results["time/rfm_stage1_cache_load_seconds"] = stage1_cache_info[
        "stage1_cache_load_seconds"
    ]
    hv_results["time/rfm_stage1_cache_compute_seconds"] = stage1_cache_info[
        "stage1_cache_compute_seconds"
    ]
    hv_results["time/rfm_stage2_cache_load_seconds"] = stage2_cache_info[
        "stage2_cache_load_seconds"
    ]
    hv_results["time/rfm_stage2_cache_compute_seconds"] = stage2_cache_info[
        "stage2_cache_compute_seconds"
    ]

    hv_results["time/rfm_offline_compute_seconds"] = (
        stage1_cache_info["stage1_cache_compute_seconds"]
        + stage2_cache_info["stage2_cache_compute_seconds"]
    )
    hv_results["time/rfm_cache_load_seconds"] = (
        stage1_cache_info["stage1_cache_load_seconds"]
        + stage2_cache_info["stage2_cache_load_seconds"]
    )

    # Important: excludes U/eigenvector/projector computation.
    hv_results["time/rfm_steered_sampling_seconds"] = steered_sampling_seconds
    hv_results["time/rfm_ode_decoding_projected_guidance_seconds"] = (
        steered_sampling_seconds
    )
    hv_results["time/rfm_evaluation_seconds"] = eval_seconds
    hv_results["time/rfm_total_wall_seconds"] = time() - total_t0

    hv_results["rfm/cache_stage1_hit"] = bool(stage1_cache_info["stage1_cache_hit"])
    hv_results["rfm/cache_stage2_hit"] = bool(stage2_cache_info["stage2_cache_hit"])
    hv_results["rfm/cache_stage1_path"] = stage1_cache_info["stage1_cache_path"]
    hv_results["rfm/cache_stage2_path"] = stage2_cache_info["stage2_cache_path"]
    hv_results["rfm/cache_dir"] = str(cache_dir)

    hv_results["rfm/sampling_variant"] = "adaptive_projected_guidance"
    hv_results["rfm/stage1_steering_family"] = stage1_steering_mode
    hv_results["rfm/stage1_steering_mode"] = (
        "adaptive_hard_gap"
        if stage1_steering_mode == "adaptive_hard"
        else ("soft_low_rank_svd_Mw" if int(D) > 2000 else "soft_full_rank")
    )
    hv_results["rfm/stage2_projector_family"] = apg_stage2_projector_mode
    hv_results["rfm/stage2_mode"] = (
        "adaptive_hard_gap"
        if apg_stage2_projector_mode == "adaptive_hard"
        else ("soft_low_rank_svd_Mw" if int(D) > 2000 else "soft_full_rank")
    )
    hv_results["rfm/stage2_used"] = True
    hv_results["rfm/guidance_gradient_location"] = "current_state_xt"
    hv_results["rfm/guidance_start_t"] = float(guidance_start_t)
    hv_results["rfm/guidance_alpha_max"] = float(guidance_alpha_max)
    hv_results["rfm/manual_subspace_k_used"] = False
    hv_results["rfm/norm_match_beta"] = 0.0
    hv_results["rfm/soft_rfm_gamma"] = (
        float(soft_rfm_gamma) if stage1_steering_mode == "soft_spectral" else 0.0
    )
    hv_results["rfm/soft_rfm_rank"] = (
        int(U_stage1.shape[-1]) if stage1_steering_mode == "soft_spectral" else 0
    )
    hv_results["rfm/stage2_soft_rfm_gamma"] = (
        float(soft_rfm_gamma) if apg_stage2_projector_mode == "soft_spectral" else 0.0
    )
    hv_results["rfm/stage2_soft_rfm_rank"] = (
        int(U_stage2_projector.shape[-1])
        if apg_stage2_projector_mode == "soft_spectral"
        else 0
    )

    hv_results["rfm/metric_weight_mode"] = str(metric_weight_mode_resolved)
    hv_results["rfm/metric_n_weights"] = int(metric_n_weights_resolved)
    hv_results["rfm/metric_same_as_sampling"] = bool(metric_same_as_sampling)
    hv_results["rfm/sampling_weight_mode"] = str(weight_mode)
    hv_results["rfm/sampling_n_weights"] = int(n_weights)

    hv_results["rfm/stage2_dataset_weight_source"] = "W_metric_cache"
    hv_results["rfm/stage2_dataset_strength"] = float(stage2_dataset_strength)
    hv_results["rfm/stage2_cache_stage1_steering_family"] = stage1_steering_mode
    hv_results["rfm/stage2_dataset_basis_num_weights"] = int(
        U_stage1_for_dataset.shape[0]
    )

    hv_results["rfm/low_rank_active"] = bool(int(D) > 2000)
    hv_results["rfm/low_rank_backend"] = (
        "svd_lowrank_Mw" if int(D) > 2000 else "exact_full_svd_Mw"
    )
    hv_results["rfm/low_rank_k"] = int(low_rank_k)
    hv_results["rfm/low_rank_niter"] = int(low_rank_niter)
    hv_results["rfm/stage1_basis_rank_max"] = int(U_stage1.shape[-1])
    hv_results["rfm/stage2_basis_rank_max"] = int(U_stage2_projector.shape[-1])
    hv_results["eval/selection_mode"] = str(eval_selection_mode)
    hv_results["eval/num_solutions"] = int(num_solutions)

    out = {
        "hv_results": hv_results,
        "X_denorm": X_denorm,
        "Y_raw": Y_raw,
        # Stage-1 noise-steering bases and optional soft spectral scales.
        "U_list": [
            U_stage1[
                k,
                :,
                : _rank_from_info(
                    info_stage1[k],
                    U_stage1[k],
                ),
            ]
            .detach()
            .cpu()
            .numpy()
            for k in range(U_stage1.shape[0])
        ],
        "U_stage1_padded": U_stage1.detach().cpu().numpy(),
        "rho_stage1": (
            rho_stage1.detach().cpu().numpy() if rho_stage1 is not None else None
        ),
        "rfm_info": info_stage1,
        "M_final": None if int(D) > 2000 else M_stage1.detach().cpu().numpy(),
        # Stage-2 projected-guidance bases and optional soft spectral scales.
        # U_stage2_projector is zero-padded to the maximum active rank;
        # U_stage2_list stores only each direction's active columns.
        "U_stage2_list": [
            U_stage2_projector[
                k,
                :,
                : _rank_from_info(
                    info_stage2[k],
                    U_stage2_projector[k],
                ),
            ]
            .detach()
            .cpu()
            .numpy()
            for k in range(U_stage2_projector.shape[0])
        ],
        "U_stage2_projector": U_stage2_projector.detach().cpu().numpy(),
        "rho_stage2_projector": (
            rho_stage2_projector.detach().cpu().numpy()
            if rho_stage2_projector is not None
            else None
        ),
        "rfm_stage2_info": info_stage2,
        "M_stage2_final": None if int(D) > 2000 else M_stage2.detach().cpu().numpy(),
        "U_stage1_for_stage2_dataset": (
            U_stage1_for_dataset.detach().cpu().numpy()
            if U_stage1_for_dataset is not None
            else None
        ),
        "rho_stage1_for_stage2_dataset": (
            rho_stage1_for_dataset.detach().cpu().numpy()
            if rho_stage1_for_dataset is not None
            else None
        ),
        "rfm_stage1_dataset_info": info_stage1_for_dataset,
        # Sampling-time preference weights.
        "W": W_sample.detach().cpu().numpy(),
    }

    return out
