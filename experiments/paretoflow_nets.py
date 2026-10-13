"""This module contains the neural network for the paretoflow-Guided Flows in
Multi-Objective Optimization."""

import math
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from offline_moo.off_moo_bench.evaluation.metrics import hv
from off_moo_bench.problem.dtlz import DTLZ
from off_moo_bench.problem.synthetic_func import SyntheticProblem
from offline_moo.off_moo_bench.utils import get_N_nondominated_indices
from offline_moo.utils import get_quantile_solutions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm
from experiments.paretoflow_utils import out_of_bounds_ratio_in_normalized_space
from offline_moo.utils import spread
from experiments.paretoflow_utils import get_reference_directions
from experiments.crd_solver import solve_simplex_qp as _solve_simplex_qp

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeEmbedding(nn.Module):
    """
    Fixed-width time embedding.

    Input:
        t: (..., 1)

    Output:
        emb: (..., out_dim)

    For the new VectorFieldNet, out_dim is hidden width M,
    not the data dimension D.

    This is MORL-friendly because the time MLP does not scale with D.
    """

    def __init__(
        self,
        D,
        out_dim=None,
        fourier_dim: int = 128,
        hidden: int = 256,
        max_freq: float = 1000.0,
    ):
        super().__init__()

        self.D = int(D)
        self.out_dim = int(D if out_dim is None else out_dim)

        # Fixed Fourier width, independent of D.
        Fdim = max(1, int(fourier_dim) // 2)
        self.Fdim = Fdim

        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(float(max_freq)),
                Fdim,
            )
        )
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(2 * Fdim + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), self.out_dim),
        )

    def forward(self, t):
        """
        Args:
            t: (..., 1)

        Returns:
            emb: (..., out_dim)
        """
        freqs = self.freqs.view(*([1] * (t.ndim - 1)), -1)
        wt = t * freqs

        emb = torch.cat(
            [
                t,
                torch.sin(wt),
                torch.cos(wt),
            ],
            dim=-1,
        )

        return self.mlp(emb)


class _FiLMSwiGLUResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult

        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)  # produces (gamma, beta)

        self.fc1 = nn.Linear(dim, 2 * inner)  # gated
        self.fc2 = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

        # ReZero: start close to identity; learn residual strength
        self.alpha = nn.Parameter(torch.zeros(1))

        # Small init on FiLM helps avoid early blow-ups
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, cond):
        # cond: (..., cond_dim)
        gamma_beta = self.film(cond)  # (..., 2*dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # (..., dim), (..., dim)

        y = self.norm(h)
        y = y * (1.0 + gamma) + beta  # FiLM modulation

        y = self.fc1(y)
        a, b = y.chunk(2, dim=-1)
        y = F.silu(a) * b  # SwiGLU
        y = self.dropout(y)
        y = self.fc2(y)

        return h + self.alpha * y


class _ResidualLinearBlock(nn.Module):
    """
    Simple residual MLP block:

        h <- h + scale * SiLU(Linear(LayerNorm(h)))

    This block uses one Linear layer. Therefore the full VectorFieldNet below
    has exactly 5 Linear layers:
        input_proj + 3 residual-block linears + out_proj.
    """

    def __init__(self, dim: int, residual_scale: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.residual_scale = float(residual_scale)

    def forward(self, h):
        return h + self.residual_scale * self.act(self.linear(self.norm(h)))


class VectorFieldNet(nn.Module):
    """
    Flow-matching vector field network.

    Final architecture:

        x in R^D
            -> x_proj: D -> M

        t in R
            -> TimeEmbedding: 1 -> M

        h = x_proj(x) + e(t)

        h
            -> LayerNorm + SiLU
            -> residual Linear blocks
            -> LayerNorm
            -> out_proj: M -> D

    Design choices:
        - no FiLM;
        - no ReZero;
        - no out_scale;
        - no internal objective/preference conditioning;
        - time is added after projecting x into hidden space;
        - hidden/time dimension does not scale with D.

    Linear depth:
        x_proj       1
        res_blocks   4
        out_proj     1
        ----------------
        total        6 Linear layers
    """

    def __init__(self, D, M=512):
        super(VectorFieldNet, self).__init__()

        self.D = int(D)
        self.M = int(M)

        # FlowMatching will construct TimeEmbedding(D, out_dim=self.time_dim).
        # This keeps time embedding width independent of data dimension D.
        self.time_dim = self.M

        # High-dimensional input projection.
        self.x_proj = nn.Linear(self.D, self.M)

        self.input_norm = nn.LayerNorm(self.M)
        self.input_act = nn.SiLU()

        n_res_blocks = 4
        residual_scale = 1.0 / math.sqrt(float(n_res_blocks))

        self.res_blocks = nn.ModuleList(
            [
                _ResidualLinearBlock(
                    self.M,
                    residual_scale=residual_scale,
                )
                for _ in range(n_res_blocks)
            ]
        )

        self.out_norm = nn.LayerNorm(self.M)
        self.out_proj = nn.Linear(self.M, self.D)

    def forward(self, x, t_emb=None):
        """
        Args:
            x:     state, shape (..., D)
            t_emb: time embedding, shape (..., M)

        Returns:
            v: velocity, shape (..., D)
        """
        if t_emb is None:
            t_emb = torch.zeros(
                *x.shape[:-1],
                self.time_dim,
                device=x.device,
                dtype=x.dtype,
            )

        if t_emb.shape[-1] != self.time_dim:
            raise ValueError(
                f"Expected t_emb.shape[-1] == {self.time_dim}, "
                f"but got t_emb.shape={tuple(t_emb.shape)}."
            )

        h = self.x_proj(x)

        # PCD-style time injection:
        # project x to hidden space, then add fixed-width time embedding.
        h = h + t_emb

        h = self.input_act(self.input_norm(h))

        for block in self.res_blocks:
            h = block(h)

        v = self.out_proj(self.out_norm(h))

        return v


class FlowMatching(nn.Module):
    def __init__(self, vnet, sigma, D, T, stochastic_euler=False, prob_path="icfm"):
        super(FlowMatching, self).__init__()

        self.vnet = vnet

        self.time_embedding = TimeEmbedding(
            D,
            out_dim=getattr(vnet, "time_dim", D),
        )

        # other params
        self.D = D

        self.T = T

        self.sigma = sigma

        self.stochastic_euler = stochastic_euler

        assert prob_path in [
            "icfm",
            "fm",
        ], (
            f"Error: The probability path could be either Independent CFM (icfm) "
            f"or Lipman's Flow Matching (fm) but {prob_path} was provided."
        )
        self.prob_path = prob_path

        self.PI = torch.from_numpy(np.asarray(np.pi))

    def log_p_base(self, x, reduction="sum", dim=1):
        log_p = -0.5 * torch.log(2.0 * self.PI) - 0.5 * x**2.0
        if reduction == "mean":
            return torch.mean(log_p, dim)
        elif reduction == "sum":
            return torch.sum(log_p, dim)
        else:
            return log_p

    def sample_base(self, x_1):
        # Gaussian base distribution
        if self.prob_path == "icfm":
            return torch.randn_like(x_1)
        elif self.prob_path == "fm":
            return torch.randn_like(x_1)
        else:
            return None

    def sample_p_t(self, x_0, x_1, t):
        if self.prob_path == "icfm":
            mu_t = (1.0 - t) * x_0 + t * x_1
            sigma_t = self.sigma
        elif self.prob_path == "fm":
            mu_t = t * x_1
            sigma_t = t * self.sigma - t + 1.0

        x = mu_t + sigma_t * torch.randn_like(x_1)

        return x

    def conditional_vector_field(self, x, x_0, x_1, t):
        if self.prob_path == "icfm":
            u_t = x_1 - x_0
        elif self.prob_path == "fm":
            u_t = (x_1 - (1.0 - self.sigma) * x) / (1.0 - (1.0 - self.sigma) * t)

        return u_t

    def forward(self, x_1, reduction="mean"):
        # =====Flow Matching
        # =====
        # z ~ q(z), e.g., q(z) = q(x_0) q(x_1), q(x_0) = base, q(x_1) = empirical
        # t ~ Uniform(0, 1)
        x_0 = self.sample_base(
            x_1
        )  # sample from the base distribution (e.g., Normal(0,I))
        t = torch.rand(size=(x_1.shape[0], 1)).to(x_1.device)

        # =====
        # sample from p(x|z)
        x = self.sample_p_t(x_0, x_1, t)  # sample independent rv

        # =====
        # invert interpolation, i.e., calculate vector field v(x,t)
        t_embd = self.time_embedding(t)
        v = self.vnet(x, t_embd)

        # =====
        # conditional vector field
        u_t = self.conditional_vector_field(x, x_0, x_1, t)

        # =====LOSS: Flow Matching
        FM_loss = torch.pow(v - u_t, 2).mean(-1)

        # Final LOSS
        if reduction == "sum":
            loss = FM_loss.sum()
        else:
            loss = FM_loss.mean()

        return loss

    @torch.no_grad()
    def sample_segment(
        self,
        x_start: torch.Tensor,  # (B, D)
        t_start: float = 0.0,
        t_end: float = 1.0,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrate unconditional ODE from t_start -> t_end starting at x_start.
        This is the same Euler sampler as sample_from_noise, but over an arbitrary sub-interval.
        """
        t_start = float(t_start)
        t_end = float(t_end)

        assert 0.0 <= t_start <= 1.0, f"t_start must be in [0,1], got {t_start}"
        assert 0.0 <= t_end <= 1.0, f"t_end must be in [0,1], got {t_end}"
        assert t_end >= t_start, f"Need t_end >= t_start, got {t_start} -> {t_end}"

        if t_end == t_start:
            return x_start

        x_t = x_start
        dtype = x_t.dtype
        dev = x_t.device
        B = x_t.shape[0]

        need_repair = repair and (xl is not None) and (xu is not None)
        if need_repair:
            xl = torch.as_tensor(xl, device=dev, dtype=dtype)
            xu = torch.as_tensor(xu, device=dev, dtype=dtype)
            if xl.ndim == 1:
                xl = xl.unsqueeze(0)
            if xu.ndim == 1:
                xu = xu.unsqueeze(0)
            if xl.shape[0] == 1:
                xl = xl.repeat(B, 1)
            if xu.shape[0] == 1:
                xu = xu.repeat(B, 1)

        # Keep roughly the same discretization density as the full sampler
        n_steps = max(1, int(round((self.T - 1) * (t_end - t_start))))
        ts = torch.linspace(t_start, t_end, n_steps + 1, device=dev, dtype=dtype)
        delta_t = ts[1] - ts[0]

        for t in ts[1:]:
            t_float = float(t)

            t_tensor = torch.tensor([[t_float]], device=dev, dtype=dtype)  # (1,1)
            t_emb = self.time_embedding(t_tensor).expand(B, -1)  # (B,D)

            v = self.vnet(x_t, t_emb)
            x_t = x_t + v * delta_t

            if need_repair and t_float >= float(t_threshold):
                x_t = self.repair_boundary(x_t, t_float, xl, xu)

        return x_t

    @torch.no_grad()
    def sample_from_noise(
        self,
        z0: torch.Tensor,  # (B, D)
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Integrate unconditional ODE from t=0 -> 1 starting at z0.
        """
        return self.sample_segment(
            x_start=z0,
            t_start=0.0,
            t_end=1.0,
            xl=xl,
            xu=xu,
            repair=repair,
            t_threshold=t_threshold,
        )

    @torch.no_grad()
    def sample_segment_rfm_projected_guided(
        self,
        x_start: torch.Tensor,  # (B, D)
        t_start: float,
        t_end: float,
        classifiers,
        weights: torch.Tensor,  # (B, m) or (m,)
        projector_basis: torch.Tensor,  # (B, D, r_max) or (D, r); zero-padded columns allowed
        projector_scales: torch.Tensor = None,
        alpha_max: float = 2.0,
        norm_match_enable: bool = False,
        norm_match_beta: float = 0.25,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Projected proxy-guided ODE integration on [t_start, t_end]:

            x_{t+dt} = x_t + dt * [v_theta(x_t, t) + alpha(t) * P_w grad_x h_w(x_t)]

        where
            alpha(t) = alpha_max * (t - t_start) / (1 - t_start),
            h_w(x_t) = sum_a w_a * (-proxy_a(x_t)),
            Hard mode:
                P_w grad = U_w (U_w^T grad)

            Soft full-rank mode:
                P_w grad = U_w diag(rho_w) (U_w^T grad)

            In soft full-rank mode, U_w has all D eigenvectors and rho_w
            contains all D spectral weights.

        This method is intended only for the two-stage RFM path. The projector
        U_w comes from the stage-2 RFM metric at t_start.
        """
        t_start = float(t_start)
        t_end = float(t_end)

        assert 0.0 <= t_start <= 1.0, f"t_start must be in [0,1], got {t_start}"
        assert 0.0 <= t_end <= 1.0, f"t_end must be in [0,1], got {t_end}"
        assert t_end >= t_start, f"Need t_end >= t_start, got {t_start} -> {t_end}"

        if t_end == t_start:
            return x_start

        x_t = x_start
        dtype = x_t.dtype
        dev = x_t.device
        B, D = x_t.shape

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(B, -1)
        weights = weights.to(device=dev, dtype=dtype)

        U = projector_basis.to(device=dev, dtype=dtype)
        if U.ndim == 2:
            U = U.unsqueeze(0).expand(B, -1, -1)

        assert (
            U.ndim == 3
        ), f"projector_basis must be (B,D,r) or (D,r), got {tuple(U.shape)}"
        assert U.shape[0] == B and U.shape[1] == D, (
            f"projector_basis must match x_start batch/dim: "
            f"got U={tuple(U.shape)}, x={tuple(x_t.shape)}"
        )

        r = int(U.shape[2])

        if projector_scales is not None:
            projector_scales = projector_scales.to(device=dev, dtype=dtype)

            if projector_scales.ndim == 1:
                projector_scales = projector_scales.view(1, -1).expand(B, -1)

            assert projector_scales.ndim == 2, (
                "projector_scales must be None, (r,), or (B,r), "
                f"got shape={tuple(projector_scales.shape)}."
            )
            assert projector_scales.shape[0] == B, (
                "projector_scales batch mismatch: "
                f"got scales={tuple(projector_scales.shape)}, B={B}."
            )
            assert projector_scales.shape[1] == r, (
                "projector_scales rank mismatch: "
                f"got scales={tuple(projector_scales.shape)}, U={tuple(U.shape)}."
            )

        need_repair = repair and (xl is not None) and (xu is not None)
        if need_repair:
            xl = torch.as_tensor(xl, device=dev, dtype=dtype)
            xu = torch.as_tensor(xu, device=dev, dtype=dtype)
            if xl.ndim == 1:
                xl = xl.unsqueeze(0)
            if xu.ndim == 1:
                xu = xu.unsqueeze(0)
            if xl.shape[0] == 1:
                xl = xl.repeat(B, 1)
            if xu.shape[0] == 1:
                xu = xu.repeat(B, 1)

        n_steps = max(1, int(round((self.T - 1) * (t_end - t_start))))
        ts = torch.linspace(t_start, t_end, n_steps + 1, device=dev, dtype=dtype)
        delta_t = ts[1] - ts[0]

        denom_sched = max(1.0 - t_start, eps)
        repair_start = float(t_threshold)

        for t in ts[1:]:
            t_float = float(t)

            # 1) Base flow velocity. No gradients through v_theta across ODE steps.
            with torch.no_grad():
                t_tensor = torch.tensor([[t_float]], device=dev, dtype=dtype)
                t_emb = self.time_embedding(t_tensor).expand(B, -1)
                v = self.vnet(x_t, t_emb)

            # 2) Full proxy gradient wrt current state x_t.
            with torch.enable_grad():
                x_leaf = x_t.detach().requires_grad_(True)

                score_list = []
                for clf in classifiers:
                    clf_dtype = next(clf.parameters()).dtype
                    x_in = (
                        x_leaf
                        if x_leaf.dtype == clf_dtype
                        else x_leaf.to(dtype=clf_dtype)
                    )
                    y_hat = clf(x_in)
                    y_hat = y_hat.view(B) if y_hat.ndim > 1 else y_hat
                    score_list.append((-1.0) * y_hat.to(device=dev, dtype=dtype))

                scores = torch.stack(score_list, dim=1)  # (B, m)
                h_w = (scores * weights).sum(dim=1).sum()  # scalar

                grad_x = torch.autograd.grad(
                    h_w,
                    x_leaf,
                    retain_graph=False,
                    create_graph=False,
                )[
                    0
                ]  # (B, D)

            coeff = torch.einsum("bdr,bd->br", U, grad_x.detach())

            if projector_scales is not None:
                coeff = coeff * projector_scales

            grad_proj = torch.einsum("bdr,br->bd", U, coeff)

            if bool(norm_match_enable):
                v_norm = v.norm(dim=1, keepdim=True)
                g_norm = grad_proj.norm(dim=1, keepdim=True)

                alpha = float(norm_match_beta) * v_norm / (g_norm + eps)
                alpha = torch.clamp(alpha, max=float(alpha_max))
            else:
                alpha_t = float(alpha_max) * max(
                    0.0,
                    (t_float - t_start) / denom_sched,
                )
                alpha = torch.full(
                    (B, 1),
                    float(alpha_t),
                    device=dev,
                    dtype=dtype,
                )

            x_t = x_t + delta_t * (v + alpha * grad_proj)

            if need_repair and t_float >= repair_start:
                x_t = self.repair_boundary(x_t, t_float, xl, xu)

        return x_t

    @torch.no_grad()
    def sample_rfm_data_adaptive_manifold_guided_knn(
        self,
        x_start: torch.Tensor,
        guidance_start_t: float,
        classifiers,
        weights: torch.Tensor,
        rfm_basis: torch.Tensor,
        rfm_scales: torch.Tensor,
        alpha_clip: float = 8.0,
        norm_match_beta: float = 0.25,
        tangent_start_t: float = 0.8,
        tangent_tau: float = 0.2,
        tangent_k_nn: int = 12,
        tangent_refresh_every: int = 5,
        knn_refresh_every: int = 15,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
        eps: float = 1e-12,
        log_tag: str = "",
        verbose_tangent_log: bool = True,
        return_stats: bool = False,
    ):
        """
        Some notes for understanding --> Data-adaptive manifold-constrained guidance, probe-free, kNN-local
        tangent, for batches that mix preference directions
        (e.g. 256 preferences x 2 instances).

        Operator unchanged:  g -> P_T R_w g,  gradient through
        x_hat_1 = x_t + (1 - t) v_theta(x_t, t).

        Tangent source: kNN-LOCAL batch PCA. Neighbors are taken in STATE
        space, so by Pareto continuity a sample's neighbors are the instances
        of nearby preferences -- the preference covering of the front supplies
        the local sample set. This avoids both failure modes:
          * per-w estimation (2 instances -> rank-1 nonsense), and
          * global pooling (adds the Pareto-set sweep + front curvature as
            spurious "tangent" directions -> biased projector).

        Rank/direction split for stability:
          * ONE shared rank r = median over all per-neighborhood tau-ranks;
          * per-sample DIRECTIONS from each sample's own neighborhood.

        No-op guarantee (do not remove): if the median tau-rank shows no
        separation (r >= min(k_nn, D)) or k_nn < r + 2, projection is skipped
        and guidance reduces to plain R_w g.

        Performance (vs the naive kNN version):
          1. The tangent is refreshed only every `tangent_refresh_every`
             steps (it rotates slowly at late t); set to 1 to recompute
             per-step (exact old behavior).
          2. The kNN graph is refreshed only every `knn_refresh_every`
             steps (neighbor identities change even more slowly).
          3. Local PCA uses the Gram trick: batched eigh of the
             (k_nn x k_nn) Gram matrices + one matmul lift, instead of
             (D x k_nn) SVDs. Mathematically the same subspace; avoids all
             dense decompositions in D, which is what made large-D tasks
             slow.

        Tangent logging (new):
          * verbose_tangent_log=True prints one summary line per call
            (= per seed x task if you pass log_tag=f"{task} seed{seed}"):

              [P_T|re61 seed3] guided 20 steps | projected 0 (0%),
              no-op 20 (100%) | refreshes: 4 -> certified 0
              (full_rank 3, knn_small 1, degenerate 0) |
              rank trace [(0.81, 7, 'noop:full_rank'), ...] |
              (D=7, k_nn=12, tau=0.2)

            Each rank-trace tuple is (t, median_rank, status) per tangent
            refresh. Status meanings:
              'proj'            -- rank certified; projection applied until
                                   the next refresh
              'noop:full_rank'  -- median rank hit min(k_nn, D): local
                                   geometry is full-rank at the tau-floor;
                                   P_T = I is the CORRECT output
                                   (the geometry saying no)
              'noop:knn_small'  -- failed only the k_nn < r + 2 check:
                                   neighborhoods too small to certify a
                                   rank they nearly found; raising
                                   tangent_k_nn may legitimately enable
                                   projection (the instrument saying no)
              'noop:degenerate' -- r < 1 (collapsed neighborhoods)
          * return_stats=True returns (x_final, stats_dict) instead of
            x_final; stats_dict keys: proj_steps, noop_steps,
            guided_proj_window, noop_frac, refreshes, certified_refreshes,
            rank_trace, noop_reasons, config. Use noop_frac as a column in
            the summary CSVs next to HV and wall time.
          * Backwards compatible: with return_stats=False (default) the
            return value is x_final exactly as before; logging only adds a
            print (silence it with verbose_tangent_log=False).
        """
        guidance_start_t = float(guidance_start_t)
        tangent_start_t = float(tangent_start_t)
        tangent_tau = float(tangent_tau)

        assert (
            0.0 <= guidance_start_t <= 1.0
        ), f"guidance_start_t must be in [0,1], got {guidance_start_t}"
        assert guidance_start_t <= tangent_start_t <= 1.0, (
            f"Need guidance_start_t <= tangent_start_t <= 1, got "
            f"{guidance_start_t} <= {tangent_start_t}"
        )
        assert (
            0.0 < tangent_tau < 1.0
        ), f"tangent_tau must be in (0,1), got {tangent_tau}"

        refresh_tan = max(1, int(tangent_refresh_every))
        refresh_knn = max(1, int(knn_refresh_every))

        x_t = x_start
        dtype = x_t.dtype
        dev = x_t.device
        B, D = x_t.shape

        k_nn = int(tangent_k_nn)
        assert (
            2 <= k_nn <= B - 1
        ), f"tangent_k_nn must be in [2, B-1] = [2, {B-1}], got {k_nn}"

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(B, -1)
        weights = weights.to(device=dev, dtype=dtype)

        U_rfm = rfm_basis.to(device=dev, dtype=dtype)
        if U_rfm.ndim == 2:
            U_rfm = U_rfm.unsqueeze(0).expand(B, -1, -1)

        assert (
            U_rfm.ndim == 3
        ), f"rfm_basis must be (B,D,D) or (D,D), got {tuple(U_rfm.shape)}"
        assert U_rfm.shape[0] == B and U_rfm.shape[1] == D, (
            f"rfm_basis must match x_start batch/dim, got "
            f"basis={tuple(U_rfm.shape)}, x={tuple(x_t.shape)}"
        )

        rfm_scales = rfm_scales.to(device=dev, dtype=dtype)
        if rfm_scales.ndim == 1:
            rfm_scales = rfm_scales.unsqueeze(0).expand(B, -1)

        assert (
            rfm_scales.ndim == 2
        ), f"rfm_scales must be (B,D) or (D,), got {tuple(rfm_scales.shape)}"
        assert rfm_scales.shape[0] == B and rfm_scales.shape[1] == U_rfm.shape[2], (
            f"rfm_scales shape mismatch: scales={tuple(rfm_scales.shape)}, "
            f"basis={tuple(U_rfm.shape)}"
        )

        need_repair = repair and (xl is not None) and (xu is not None)
        if need_repair:
            xl = torch.as_tensor(xl, device=dev, dtype=dtype)
            xu = torch.as_tensor(xu, device=dev, dtype=dtype)

            if xl.ndim == 1:
                xl = xl.unsqueeze(0)
            if xu.ndim == 1:
                xu = xu.unsqueeze(0)

            if xl.shape[0] == 1:
                xl = xl.repeat(B, 1)
            if xu.shape[0] == 1:
                xu = xu.repeat(B, 1)

        ts = torch.linspace(0.0, 1.0, self.T, device=dev, dtype=dtype)
        delta_t = ts[1] - ts[0]
        repair_start = float(t_threshold)

        def knn_graph(states: torch.Tensor) -> torch.Tensor:
            """Neighbor indices (B, k_nn), self excluded."""
            dist = torch.cdist(states, states)
            return dist.topk(k_nn + 1, largest=False).indices[:, 1:]

        def local_tangent_gram(states: torch.Tensor, knn_idx: torch.Tensor):
            """
            Batched Gram-trick local PCA: per-sample tangent bases, or None.

            Top-r column space of each centered neighborhood C_i (D x k_nn)
            from eigh of G_i = C_i^T C_i (k_nn x k_nn):
                G_i = V S^2 V^T  =>  U_i = C_i V_r S_r^{-1}.
            Same subspace as the (D x k_nn) SVD; no decomposition in D.

            Returns (U_loc, r, status): U_loc (B, D, r) orthonormal columns
            with status 'proj', or U_loc=None with status in
            {'noop:full_rank', 'noop:knn_small', 'noop:degenerate'}.
            r is always the median tau-rank the rule selected, so no-op
            causes are attributable from the log.
            """
            nbrs = states[knn_idx]  # (B, k, D)
            nbrs = nbrs - nbrs.mean(dim=1, keepdim=True)

            G = torch.bmm(nbrs, nbrs.transpose(1, 2))  # (B, k, k)
            evals, V = torch.linalg.eigh(G)  # ascending
            evals = evals.flip(-1).clamp(min=0.0)  # descending
            V = V.flip(-1)
            sigma = evals.sqrt()  # (B, k)

            floor = tangent_tau * sigma[:, :1].clamp(min=eps)
            per_rank = (sigma >= floor).sum(dim=1)  # (B,)
            r = int(per_rank.float().median().item())

            # No-op: no local separation, or neighborhoods too small.
            if r < 1:
                return None, r, "noop:degenerate"
            if r >= min(k_nn, D):
                return None, r, "noop:full_rank"
            if k_nn < r + 2:
                return None, r, "noop:knn_small"

            V_r = V[:, :, :r]  # (B, k, r)
            inv_s = 1.0 / sigma[:, :r].clamp(min=eps)  # (B, r)
            U_loc = torch.bmm(nbrs.transpose(1, 2), V_r) * inv_s.unsqueeze(1)
            return U_loc, r, "proj"  # (B, D, r)

        tan_U = None
        tan_step = None
        knn_idx = None
        knn_step = None

        # ---- tangent logging state ----
        proj_steps = 0  # guided steps where projection applied
        noop_steps = 0  # guided steps where it was skipped
        rank_trace = []  # (t, median_rank, status) per refresh
        noop_reasons = {"noop:full_rank": 0, "noop:knn_small": 0, "noop:degenerate": 0}
        certified_refreshes = 0

        for step_idx, t in enumerate(ts[1:], start=1):
            t_float = float(t)
            t_tensor = torch.tensor([[t_float]], device=dev, dtype=dtype)

            with torch.no_grad():
                t_emb = self.time_embedding(t_tensor).expand(B, -1)
                v = self.vnet(x_t, t_emb)

            if t_float >= guidance_start_t:
                with torch.enable_grad():
                    x_leaf = x_t.detach().requires_grad_(True)
                    t_emb_grad = self.time_embedding(t_tensor).expand(B, -1)

                    v_for_grad = self.vnet(x_leaf, t_emb_grad)
                    x_hat_1 = x_leaf + (1.0 - t_float) * v_for_grad

                    score_list = []
                    for clf in classifiers:
                        clf_dtype = next(clf.parameters()).dtype
                        x_in = (
                            x_hat_1
                            if x_hat_1.dtype == clf_dtype
                            else x_hat_1.to(dtype=clf_dtype)
                        )

                        y_hat = clf(x_in)
                        y_hat = y_hat.view(B) if y_hat.ndim > 1 else y_hat

                        score_list.append((-1.0) * y_hat.to(device=dev, dtype=dtype))

                    scores = torch.stack(score_list, dim=1)
                    h_w = (scores * weights).sum(dim=1).sum()

                    grad_x = torch.autograd.grad(
                        h_w,
                        x_leaf,
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                coeff = torch.einsum("bdr,bd->br", U_rfm, grad_x.detach())
                coeff = coeff * rfm_scales
                grad_guided = torch.einsum("bdr,br->bd", U_rfm, coeff)

                if t_float >= tangent_start_t:
                    if knn_idx is None or step_idx - knn_step >= refresh_knn:
                        knn_idx = knn_graph(x_t)
                        knn_step = step_idx

                    if tan_step is None or step_idx - tan_step >= refresh_tan:
                        tan_U, tan_rank, tan_status = local_tangent_gram(x_t, knn_idx)
                        tan_step = step_idx
                        rank_trace.append((round(t_float, 3), tan_rank, tan_status))
                        if tan_status == "proj":
                            certified_refreshes += 1
                        else:
                            noop_reasons[tan_status] += 1

                    if tan_U is not None:
                        proj_steps += 1
                        c = torch.einsum("bdr,bd->br", tan_U, grad_guided)
                        grad_guided = torch.einsum("bdr,br->bd", tan_U, c)
                    else:
                        noop_steps += 1
                        # no certified local tangent -> plain R_w guidance

                v_norm = v.norm(dim=1, keepdim=True)
                g_norm = grad_guided.norm(dim=1, keepdim=True)

                alpha = float(norm_match_beta) * v_norm / (g_norm + eps)
                alpha = torch.clamp(alpha, max=float(alpha_clip))

                x_t = x_t + delta_t * (v + alpha * grad_guided)

            else:
                x_t = x_t + delta_t * v

            if need_repair and t_float >= repair_start:
                x_t = self.repair_boundary(x_t, t_float, xl, xu)

        # ---- tangent log summary (per call = per seed x task) ----
        total_window = proj_steps + noop_steps
        noop_frac = noop_steps / total_window if total_window > 0 else 0.0

        if verbose_tangent_log:
            tag = f"|{log_tag}" if log_tag else ""
            if total_window == 0:
                print(
                    f"[P_T{tag}] tangent window empty "
                    f"(tangent_start_t={tangent_start_t} vs T={self.T} grid) "
                    f"| (D={D}, k_nn={k_nn}, tau={tangent_tau})"
                )
            else:
                print(
                    f"[P_T{tag}] guided {total_window} steps | "
                    f"projected {proj_steps} "
                    f"({100.0 * proj_steps / total_window:.0f}%), "
                    f"no-op {noop_steps} "
                    f"({100.0 * noop_frac:.0f}%) | "
                    f"refreshes: {len(rank_trace)} -> certified "
                    f"{certified_refreshes} "
                    f"(full_rank {noop_reasons['noop:full_rank']}, "
                    f"knn_small {noop_reasons['noop:knn_small']}, "
                    f"degenerate {noop_reasons['noop:degenerate']}) | "
                    f"rank trace {rank_trace} | "
                    f"(D={D}, k_nn={k_nn}, tau={tangent_tau})"
                )

        if return_stats:
            stats = {
                "proj_steps": proj_steps,
                "noop_steps": noop_steps,
                "guided_proj_window": total_window,
                "noop_frac": noop_frac,
                "refreshes": len(rank_trace),
                "certified_refreshes": certified_refreshes,
                "rank_trace": rank_trace,
                "noop_reasons": dict(noop_reasons),
                "config": {
                    "D": D,
                    "B": B,
                    "k_nn": k_nn,
                    "tau": tangent_tau,
                    "tangent_start_t": tangent_start_t,
                    "tangent_refresh_every": refresh_tan,
                    "knn_refresh_every": refresh_knn,
                },
            }
            return x_t, stats

        return x_t

    @torch.no_grad()
    def sample_rfm_data_adaptive_manifold_guided_batch(
        self,
        x_start: torch.Tensor,
        guidance_start_t: float,
        classifiers,
        weights: torch.Tensor,
        rfm_basis: torch.Tensor,
        rfm_scales: torch.Tensor,
        alpha_clip: float = 8.0,
        norm_match_beta: float = 0.25,
        tangent_start_t: float = 0.8,
        tangent_tau: float = 0.2,
        tangent_k_nn: int = 12,
        tangent_refresh_every: int = 1,
        knn_refresh_every: int = 15,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compatibility entry point for the fixed-localspace fast R1 DAMG."""
        return self.sample_rfm_data_adaptive_manifold_guided_knn(
            x_start=x_start,
            guidance_start_t=guidance_start_t,
            classifiers=classifiers,
            weights=weights,
            rfm_basis=rfm_basis,
            rfm_scales=rfm_scales,
            alpha_clip=alpha_clip,
            norm_match_beta=norm_match_beta,
            tangent_start_t=tangent_start_t,
            tangent_tau=tangent_tau,
            tangent_k_nn=tangent_k_nn,
            tangent_refresh_every=tangent_refresh_every,
            knn_refresh_every=knn_refresh_every,
            xl=xl,
            xu=xu,
            repair=repair,
            t_threshold=t_threshold,
            eps=eps,
        )

    @torch.no_grad()
    def sample_crd_reduced_guided(
        self,
        x_start: torch.Tensor,
        guidance_start_t: float,
        classifiers,
        weights: torch.Tensor,
        robust_gate_c: float = 0.1,  # NOTE: 1e-4 was calibrated for RAW-J scales;
        norm_match_beta: float = 0.25,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """CRD-Reduced: normalized-J consensus/alignment + scalar certificate.

        Pipeline per guided step:
            endpoint grads -> normalize J -> Tikhonov MGDA consensus
            -> primary alignment -> omega gate blend -> sign certificate
            -> norm-matched update (NO clip, NO RFM, NO tangent, NO neighbors).
        Guarantee: scalar effect <J_u w, g~> >= 0 .
        """
        guidance_start_t = float(guidance_start_t)
        x_t = x_start
        dtype, dev = x_t.dtype, x_t.device
        B, D = x_t.shape

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(B, -1)
        weights = weights.to(device=dev, dtype=dtype)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        m_obj = int(weights.shape[1])

        gate_c_base = max(float(robust_gate_c), float(eps))
        eps_tik = 1e-8

        need_repair = bool(repair) and (xl is not None) and (xu is not None)
        if need_repair:
            xl = torch.as_tensor(xl, device=dev, dtype=dtype)
            xu = torch.as_tensor(xu, device=dev, dtype=dtype)
            if xl.ndim == 1:
                xl = xl.unsqueeze(0)
            if xu.ndim == 1:
                xu = xu.unsqueeze(0)
            if xl.shape[0] == 1:
                xl = xl.repeat(B, 1)
            if xu.shape[0] == 1:
                xu = xu.repeat(B, 1)

        ts = torch.linspace(0.0, 1.0, self.T, device=dev, dtype=dtype)
        delta_t = ts[1] - ts[0]
        repair_start = float(t_threshold)

        diag_index = 0
        info_log = []
        self._last_crd_reduced_info_log = info_log

        # Normalized consensus QP: closed form for one/two objectives,
        # shared batched simplex solver for higher objective counts.
        def min_norm_consensus_normalized(J):
            raw_grad_norms = J.norm(dim=1).clamp_min(eps)  # (B,m)
            Ju = J / raw_grad_norms.unsqueeze(1)
            Q_eff = torch.einsum("bdm,bdn->bmn", Ju, Ju)
            Q_eff = 0.5 * (Q_eff + Q_eff.transpose(1, 2))
            tr_eff = torch.diagonal(Q_eff, dim1=1, dim2=2).sum(dim=1)
            tik_scale = eps_tik * tr_eff / float(m_obj)
            I = torch.eye(m_obj, device=dev, dtype=dtype).unsqueeze(0)
            Q = Q_eff + tik_scale.view(B, 1, 1) * I
            Q = 0.5 * (Q + Q.transpose(1, 2))
            if m_obj == 1:
                pi = torch.ones(B, 1, device=dev, dtype=dtype)
            elif m_obj == 2:
                q11, q22, q12 = Q[:, 0, 0], Q[:, 1, 1], Q[:, 0, 1]
                diff = q11 - 2.0 * q12 + q22
                pi_1 = torch.clamp((q22 - q12) / diff.clamp_min(1e-14), 0.0, 1.0)
                pi_1 = torch.where(diff < 1e-14, torch.full_like(pi_1, 0.5), pi_1)
                pi = torch.stack([pi_1, 1.0 - pi_1], dim=1)
            else:
                pi = _solve_simplex_qp(Q)
            d_cons = torch.einsum("bdm,bm->bd", Ju, pi)
            return d_cons, pi, Ju

        def endpoint_scores_and_J(x_leaf, x_eval):
            score_list = []
            for clf in classifiers:
                clf_dtype = next(clf.parameters()).dtype
                x_in = (
                    x_eval if x_eval.dtype == clf_dtype else x_eval.to(dtype=clf_dtype)
                )
                y_hat = clf(x_in)
                y_hat = y_hat.view(B) if y_hat.ndim > 1 else y_hat
                score_list.append((-1.0) * y_hat.to(device=dev, dtype=dtype))
            scores = torch.stack(score_list, dim=1)  # (B,m)
            grad_cols = []
            for obj_idx in range(m_obj):
                grad_i = torch.autograd.grad(
                    scores[:, obj_idx].sum(),
                    x_leaf,
                    retain_graph=obj_idx < (m_obj - 1),
                    create_graph=False,
                )[0]
                grad_cols.append(grad_i)
            J = torch.stack(grad_cols, dim=2)  # (B,D,m)
            J = torch.nan_to_num(J.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            scores = torch.nan_to_num(scores.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            return scores, J

        for step_idx, t in enumerate(ts[1:], start=1):
            t_float = float(t)
            t_tensor = torch.tensor([[t_float]], device=dev, dtype=dtype)
            is_guided_step = t_float >= guidance_start_t

            if is_guided_step:
                with torch.enable_grad():
                    x_leaf = x_t.detach().requires_grad_(True)
                    t_emb = self.time_embedding(t_tensor).expand(B, -1)
                    v_leaf = self.vnet(x_leaf, t_emb)
                    x_hat_1 = x_leaf + (1.0 - t_float) * v_leaf
                    scores, J = endpoint_scores_and_J(x_leaf, x_hat_1)
                v = v_leaf.detach()

                d_cons, pi_star, Ju = min_norm_consensus_normalized(J)
                d_cons_norm = d_cons.norm(dim=1, keepdim=True)
                d_cons_unit = d_cons / (d_cons_norm + eps)

                score_shift = scores - scores.min(dim=1, keepdim=True).values
                score_profile = score_shift + eps
                score_profile = score_profile / (
                    score_profile.sum(dim=1, keepdim=True) + eps
                )

                d_align_raw = torch.einsum("bdm,bm->bd", Ju, weights - score_profile)
                d_align = d_align_raw / (d_align_raw.norm(dim=1, keepdim=True) + eps)
                d_align = torch.nan_to_num(d_align, nan=0.0, posinf=0.0, neginf=0.0)

                omega = d_cons_norm.pow(2) / (d_cons_norm.pow(2) + gate_c_base)
                g_tilde = omega * d_cons_unit + (1.0 - omega) * d_align
                g_tilde = torch.nan_to_num(g_tilde, nan=0.0, posinf=0.0, neginf=0.0)

                # ---- scalar sign certificate ----
                g_w = torch.einsum("bdm,bm->bd", Ju, weights)  # safe direction
                scalar_eff = (g_w * g_tilde).sum(dim=1)
                cert_fired = scalar_eff < 0.0
                if bool(cert_fired.any()):
                    g_tilde = torch.where(cert_fired.view(B, 1), g_w, g_tilde)

                # ---- norm-matched update, NO clip ----
                v_norm = v.norm(dim=1, keepdim=True)
                g_norm = g_tilde.norm(dim=1, keepdim=True)
                alpha = float(norm_match_beta) * v_norm / (g_norm + eps)
                x_t = x_t + delta_t * (v + alpha * g_tilde)

                b = int(diag_index)
                per_obj_eff = torch.einsum("bdm,bd->bm", Ju, g_tilde)
                info_log.append(
                    {
                        "t": t_float,
                        "pi_star": pi_star[b].detach().cpu().numpy(),
                        "omega": float(omega[b, 0].detach().cpu()),
                        "certificate_fired": bool(cert_fired[b].detach().cpu()),
                        "scalar_effect": float(
                            (weights[b] * per_obj_eff[b]).sum().detach().cpu()
                        ),
                        "per_objective_effect": per_obj_eff[b].detach().cpu().numpy(),
                        "num_negative_per_objective": int(
                            (per_obj_eff[b] < 0.0).sum().detach().cpu()
                        ),
                        "alpha": float(alpha[b, 0].detach().cpu()),
                    }
                )
            else:
                with torch.no_grad():
                    t_emb = self.time_embedding(t_tensor).expand(B, -1)
                    v = self.vnet(x_t, t_emb)
                x_t = x_t + delta_t * v

            if need_repair and t_float >= repair_start:
                x_t = self.repair_boundary(x_t, t_float, xl, xu)

        return x_t

    @torch.no_grad()
    def sample_cone_projected_guided(
        self,
        x_start: torch.Tensor,
        guidance_start_t: float,
        classifiers,
        weights: torch.Tensor,
        norm_match_beta: float = 0.25,
        xl=None,
        xu=None,
        repair: bool = False,
        t_threshold: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Cone-projected guidance with a min-norm fallback.

        Pipeline per guided step:
            endpoint grads -> normalize J -> preferred direction
            u_pref = Ju(w - score_profile) + Ju w
            -> project onto K = {s : <Ju_a, s> >= 0 for all a}
            -> min-norm fallback if projection is ~0 (Pareto-stationary)
            -> norm-matched update (NO clip).
        Guarantee (Theorem 1), unconditionally, hence for every simplex weighting.
        Hyperparameters: guidance_start_t, norm_match_beta. No carried state.
        """
        guidance_start_t = float(guidance_start_t)
        x_t = x_start
        dtype, dev = x_t.dtype, x_t.device
        B, D = x_t.shape

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(B, -1)
        weights = weights.to(device=dev, dtype=dtype)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
        m_obj = int(weights.shape[1])
        eps_tik = 1e-8
        feas_tol = 1e-9  # constraint feasibility tolerance in the active-set search

        need_repair = bool(repair) and (xl is not None) and (xu is not None)
        if need_repair:
            xl = torch.as_tensor(xl, device=dev, dtype=dtype)
            xu = torch.as_tensor(xu, device=dev, dtype=dtype)
            if xl.ndim == 1:
                xl = xl.unsqueeze(0)
            if xu.ndim == 1:
                xu = xu.unsqueeze(0)
            if xl.shape[0] == 1:
                xl = xl.repeat(B, 1)
            if xu.shape[0] == 1:
                xu = xu.repeat(B, 1)

        ts = torch.linspace(0.0, 1.0, self.T, device=dev, dtype=dtype)
        delta_t = ts[1] - ts[0]
        repair_start = float(t_threshold)

        diag_index = 0
        info_log = []
        self._last_cone_info_log = info_log

        def endpoint_scores_and_J(x_leaf, x_eval):
            score_list = []
            for clf in classifiers:
                clf_dtype = next(clf.parameters()).dtype
                x_in = (
                    x_eval if x_eval.dtype == clf_dtype else x_eval.to(dtype=clf_dtype)
                )
                y_hat = clf(x_in)
                y_hat = y_hat.view(B) if y_hat.ndim > 1 else y_hat
                score_list.append((-1.0) * y_hat.to(device=dev, dtype=dtype))
            scores = torch.stack(score_list, dim=1)
            grad_cols = []
            for obj_idx in range(m_obj):
                grad_i = torch.autograd.grad(
                    scores[:, obj_idx].sum(),
                    x_leaf,
                    retain_graph=obj_idx < (m_obj - 1),
                    create_graph=False,
                )[0]
                grad_cols.append(grad_i)
            J = torch.stack(grad_cols, dim=2)
            J = torch.nan_to_num(J.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            scores = torch.nan_to_num(scores.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            return scores, J

        def min_norm_point(Ju):
            """MGDA min-norm point on Tikhonov-regularized normalized Gram."""
            Q_eff = torch.einsum("bdm,bdn->bmn", Ju, Ju)
            Q_eff = 0.5 * (Q_eff + Q_eff.transpose(1, 2))
            tr_eff = torch.diagonal(Q_eff, dim1=1, dim2=2).sum(dim=1)
            I = torch.eye(m_obj, device=dev, dtype=dtype).unsqueeze(0)
            Q = Q_eff + (eps_tik * tr_eff / float(m_obj)).view(B, 1, 1) * I
            Q = 0.5 * (Q + Q.transpose(1, 2))
            if m_obj == 2:
                q11, q22, q12 = Q[:, 0, 0], Q[:, 1, 1], Q[:, 0, 1]
                diff = q11 - 2.0 * q12 + q22
                pi_1 = torch.clamp((q22 - q12) / diff.clamp_min(1e-14), 0.0, 1.0)
                pi_1 = torch.where(diff < 1e-14, torch.full_like(pi_1, 0.5), pi_1)
                pi = torch.stack([pi_1, 1.0 - pi_1], dim=1)
            else:
                pi = _solve_simplex_qp(Q)
            return torch.einsum("bdm,bm->bd", Ju, pi)

        def cone_project(u, Ju):
            """Batched exact projection of u onto K = {s : Ju^T s >= 0}.

            KKT: s* = u + Ju mu with mu >= 0, complementary slackness.
            Enumerate the 2^m - 1 candidate active sets on the m x m Gram;
            among KKT-consistent feasible candidates take the one with the
            smallest ||s - u||. Vectorized over the batch, loop over masks
            (m <= 4 => at most 15 masks). Rows with u already in K are
            returned unchanged.
            """
            dots = torch.einsum("bdm,bd->bm", Ju, u)  # (B,m)
            inside = (dots >= -feas_tol).all(dim=1)  # (B,)
            s_out = u.clone()
            if bool(inside.all()):
                return s_out, inside

            Gram = torch.einsum("bdm,bdn->bmn", Ju, Ju)
            Gram = 0.5 * (Gram + Gram.transpose(1, 2))
            best_d2 = torch.full((B,), float("inf"), device=dev, dtype=dtype)
            best_s = torch.zeros_like(u)
            found = torch.zeros(B, device=dev, dtype=torch.bool)

            for mask in range(1, 1 << m_obj):
                idx = [a for a in range(m_obj) if (mask >> a) & 1]
                na = len(idx)
                idx_t = torch.tensor(idx, device=dev, dtype=torch.long)
                A_gram = Gram[:, idx_t][:, :, idx_t]  # (B,na,na)
                A_gram = A_gram + eps_tik * torch.eye(na, device=dev, dtype=dtype)
                rhs = -dots[:, idx_t]  # (B,na)
                try:
                    mu = torch.linalg.solve(A_gram, rhs.unsqueeze(2)).squeeze(2)
                except RuntimeError:
                    continue
                kkt_ok = (mu >= -feas_tol).all(dim=1)
                s_cand = u + torch.einsum("bdm,bm->bd", Ju[:, :, idx_t], mu)
                cand_dots = torch.einsum("bdm,bd->bm", Ju, s_cand)
                feas_ok = (cand_dots >= -1e-6).all(dim=1)
                d2 = ((s_cand - u) ** 2).sum(dim=1)
                ok = kkt_ok & feas_ok & (~inside)
                upd = ok & (d2 < best_d2)
                best_s = torch.where(upd.view(B, 1), s_cand, best_s)
                best_d2 = torch.where(upd, d2, best_d2)
                found = found | ok
            # rows outside K with a found candidate get it; unfound rows -> 0
            # (caller applies the min-norm fallback to near-zero rows)
            outside = ~inside
            s_out = torch.where((outside & found).view(B, 1), best_s, s_out)
            s_out = torch.where(
                (outside & ~found).view(B, 1), torch.zeros_like(u), s_out
            )
            return s_out, inside

        for step_idx, t in enumerate(ts[1:], start=1):
            t_float = float(t)
            t_tensor = torch.tensor([[t_float]], device=dev, dtype=dtype)
            is_guided_step = t_float >= guidance_start_t

            if is_guided_step:
                with torch.enable_grad():
                    x_leaf = x_t.detach().requires_grad_(True)
                    t_emb = self.time_embedding(t_tensor).expand(B, -1)
                    v_leaf = self.vnet(x_leaf, t_emb)
                    x_hat_1 = x_leaf + (1.0 - t_float) * v_leaf
                    scores, J = endpoint_scores_and_J(x_leaf, x_hat_1)
                v = v_leaf.detach()

                raw_grad_norms = J.norm(dim=1).clamp_min(eps)
                Ju = J / raw_grad_norms.unsqueeze(1)

                score_shift = scores - scores.min(dim=1, keepdim=True).values
                score_profile = score_shift + eps
                score_profile = score_profile / (
                    score_profile.sum(dim=1, keepdim=True) + eps
                )

                u_pref = torch.einsum(
                    "bdm,bm->bd", Ju, weights - score_profile
                ) + torch.einsum("bdm,bm->bd", Ju, weights)

                g_tilde, was_inside = cone_project(u_pref, Ju)

                # ---- Pareto-stationary fallback: min-norm point ----
                tiny = g_tilde.norm(dim=1) < 1e-8
                if bool(tiny.any()):
                    d_mn = min_norm_point(Ju)
                    g_tilde = torch.where(tiny.view(B, 1), d_mn, g_tilde)

                # ---- numerical backstop (vacuous in exact arithmetic) ----
                g_w = torch.einsum("bdm,bm->bd", Ju, weights)
                bad = (g_w * g_tilde).sum(dim=1) < 0.0
                if bool(bad.any()):
                    g_tilde = torch.where(bad.view(B, 1), g_w, g_tilde)

                # ---- norm-matched update, NO clip ----
                v_norm = v.norm(dim=1, keepdim=True)
                g_norm = g_tilde.norm(dim=1, keepdim=True)
                alpha = float(norm_match_beta) * v_norm / (g_norm + eps)
                x_t = x_t + delta_t * (v + alpha * g_tilde)

                b = int(diag_index)
                per_obj_eff = torch.einsum("bdm,bd->bm", Ju, g_tilde)
                info_log.append(
                    {
                        "t": t_float,
                        "u_pref_inside_cone": bool(was_inside[b].detach().cpu()),
                        "fallback_min_norm": bool(tiny[b].detach().cpu()),
                        "backstop_fired": bool(bad[b].detach().cpu()),
                        "per_objective_effect": per_obj_eff[b].detach().cpu().numpy(),
                        "num_negative_per_objective": int(
                            (per_obj_eff[b] < -1e-6).sum().detach().cpu()
                        ),  # float32 boundary tol
                        "raw_norm_ratio": float(
                            (
                                raw_grad_norms[b].max()
                                / raw_grad_norms[b].min().clamp_min(eps)
                            )
                            .detach()
                            .cpu()
                        ),
                        "alpha": float(alpha[b, 0].detach().cpu()),
                    }
                )
            else:
                with torch.no_grad():
                    t_emb = self.time_embedding(t_tensor).expand(B, -1)
                    v = self.vnet(x_t, t_emb)
                x_t = x_t + delta_t * v

            if need_repair and t_float >= repair_start:
                x_t = self.repair_boundary(x_t, t_float, xl, xu)

        return x_t

    #     Since we know u_t(x|x_1) = (x_1 - (1 - sigma) * x) / (1 - (1 - sigma) * t)
    #     We can solve for x_1

    def weighted_conditional_vnet(
        self, x_t, t, classifiers, weights, gamma=2.0, t_threshold=0.8, **kwargs
    ):
        """
        Same signature, but during guidance (t >= t_threshold) the gradient includes
        du_t/dx_t by computing u_t = vnet(x_t_ + t_emb) with grad enabled.
        """
        # Make x_t a leaf requiring grad (for guidance gradient)
        x_t_ = x_t.detach().requires_grad_(True)  # (B, D)

        # Build time embedding with consistent shape (1,1) then broadcast to (B,D)
        # (Using no_grad here is fine; we only need grad w.r.t. x_t_.)
        with torch.no_grad():
            t_tensor = torch.tensor(
                [[t]], device=x_t_.device, dtype=x_t_.dtype
            )  # (1,1)
            t_emb = self.time_embedding(t_tensor)  # (1, D)
            t_emb = t_emb.expand(x_t_.shape[0], -1)  # (B, D)

        # If we are before guidance threshold, keep the fast path (no guidance)
        if t < t_threshold:
            with torch.no_grad():
                u_t = self.vnet(x_t, t_emb)  # (B, D)
            return u_t

        # ---------- Guidance period: enable grad through u_t(x_t_) ----------
        # IMPORTANT: do NOT detach x_t_ here; we want du_t/dx_t_.
        u_t = self.vnet(x_t_, t_emb)  # (B, D), tracked w.r.t. x_t_

        # Your x1 reconstruction (kept as-is)
        x_1 = u_t * (1 - (1 - self.sigma) * t) + (1 - self.sigma) * x_t_  # (B, D)

        # Proxy scores (need grad w.r.t. x_1 -> x_t_)
        scores = []
        for classifier in classifiers:
            scores.append(-1 * classifier(x_1))  # each typically (B,) or (B,1)

        scores = torch.stack(scores, dim=1)  # (B, n_obj, ...)
        scores = scores.squeeze(-1)  # -> (B, n_obj) if needed

        # Weighted scalar objective (scalar)
        log_value = (scores * weights).sum(dim=1).sum()

        # Compute grad ONLY w.r.t. x_t_ (includes du_t/dx_t_)
        grad_x = torch.autograd.grad(
            log_value, x_t_, retain_graph=False, create_graph=False
        )[
            0
        ]  # (B, D)

        denom = max(t, kwargs["delta_t"])
        scale = gamma * (1 - float(t)) / denom

        # Detach to avoid carrying graphs across Euler steps
        return u_t.detach() + scale * grad_x.detach()

    @classmethod
    def get_neighborhood_indices(cls, weight, objectives_weights, K, distance="cosine"):
        """
        weight: the weight for the objective
        objectives_weights: the weights for each objective
        K: the number of samples we want to include in the neighborhood
        distance: the distance metric to calculate the distance
        between the weight and the weights of the objectives. The default is cosine angle
        distance, but can be changed to Euclidean distance
        return: the indices of the neighborhood samples
        """
        if distance == "cosine":
            # calculate the cosine similarity between the weight and the weights of
            # the objectives
            cos = nn.CosineSimilarity(dim=1)
            cos_similarities = cos(objectives_weights, weight.unsqueeze(0))
            # sort the cosine similarities and get the indices of the K nearest neighbors
            _, indices = torch.topk(cos_similarities, K, largest=True)
            return indices

        if distance == "euclidean":
            # calculate the distance between the weight and the weights of
            # the objectives based on the Euclidean distance
            distances = torch.norm(objectives_weights - weight, dim=1)
            # sort the distances and get the indices of the K nearest neighbors
            _, indices = torch.topk(distances, K, largest=False)
            return indices

    # Get Objectives Weights and Number of Samples we want to generate
    @classmethod
    def calculate_objectives_weights(cls, M, num_solutions):
        """
        Get weights for each objective, number of objectives is equal to number of classifiers
        n_partitions is equal to the number of samples we want to generate,
        because we want to generate a batch of
        samples to maximize different objectives, so that can maximize the hypervolume
        len(objectives_weights) = combination(M + n_partitions - 1, n_partitions)
        where M is the number of objectives

        M: the number of objectives
        num_solutions: the number of new solutions we want to keep in the final pareto set
        the number of samples needed to generate should be larger than or equal to num_solutions
        return: objective weights, the number of samples we want to generate
        """
        n_partitions = 1
        while True:
            if math.comb(M + n_partitions - 1, n_partitions) >= 500:
                break
            n_partitions += 1
        objectives_weights = get_reference_directions(
            "uniform", M, n_partitions=n_partitions
        )
        batch_size = objectives_weights.shape[0]

        # shape: (batch_size, len(classifiers))
        objectives_weights = torch.tensor(objectives_weights).to(device)

        return objectives_weights, batch_size

    @classmethod
    def calculate_angles(cls, a, b):
        """
        a: the first tensor, shape: (n, D)
        b: the second tensor, shape: (n, D)
        return: the angle between the two tensors, shape: (n, 1)
        """
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        angle = torch.acos(cos)
        return angle.unsqueeze(1)

    @classmethod
    def get_ts_and_delta_t(cls, T, t_threshold=0.8, adaptive=False):
        """
        T: the number of steps to generate the samples
        return: the time steps and the step size
        """
        if adaptive:
            ts1 = torch.linspace(0.0, t_threshold, int(T * (1 - t_threshold)))
            ts2 = torch.linspace(t_threshold, 1.0, T - int(T * (1 - t_threshold)))
            ts = torch.cat((ts1, ts2))
            assert ts.shape[0] == T
            delta_t1 = (
                ts1[1] - ts1[0] if ts1.shape[0] > 1 else 1.0
            )  # Handle the case when t_threshold = 1
            delta_t2 = (
                ts2[1] - ts2[0] if ts2.shape[0] > 1 else delta_t1
            )  # Handle the case when t_threshold = 0

            def delta_t(t):
                if t < t_threshold:
                    return delta_t1
                return delta_t2

            return ts, delta_t
        else:
            ts = torch.linspace(0.0, 1.0, T)
            d_t = ts[1] - ts[0]

            def delta_t(t):
                return d_t

            return ts, delta_t

    def initialize_pareto_set(
        self, batch_size, objectives_weights=None, task=None, methods="d_best"
    ):
        """
        batch_size: the number of samples we want to generate
        return: the pareto set of the generated samples
        """

        # Initialize the pareto set with empty values
        if methods == "empty_init":
            pareto_set = [(torch.empty(self.D), float("-inf"))] * batch_size
            return pareto_set
        # Initialize the pareto set with the existing best samples from the offline dataset
        elif methods == "d_best":
            assert task is not None, "Error: The task should be provided"
            assert (
                objectives_weights is not None
            ), "Error: The objectives_weights_list should be provided"
            assert (
                len(objectives_weights) == batch_size
            ), "Error: Length of objectives_weights_list must equal batch_size"

            # Get all solutions from the offline dataset
            all_x, all_y = task.get_N_non_dominated_solutions(
                N=batch_size, return_x=True, return_y=True
            )

            if task.is_discrete:
                all_x = task.to_logits(all_x)
            all_x = task.normalize_x(all_x)
            all_y = task.normalize_y(all_y)

            # Convert to tensors
            all_x = torch.tensor(all_x).to(device)
            all_y = torch.tensor(all_y).to(device)

            # Initialize Pareto set
            pareto_set = []
            remaining_indices = torch.arange(all_x.size(0)).to(device)

            for i in range(batch_size):
                weights = objectives_weights[i].to(
                    device
                )  # Weight vector for position i

                # Compute scalarized scores
                scalarized_scores = torch.matmul(all_y, weights)

                # Select the candidate with the best scalarized score
                best_index = torch.argmax(scalarized_scores)
                best_x = all_x[best_index]
                best_score = scalarized_scores[best_index]

                # Add to Pareto set
                pareto_set.append((best_x, best_score))

                # Remove the selected candidate from consideration
                mask = torch.ones(all_x.size(0), dtype=torch.bool).to(device)
                mask[best_index] = False
                all_x = all_x[mask]
                all_y = all_y[mask]
                remaining_indices = remaining_indices[mask]

            return pareto_set
        else:
            raise ValueError("Invalid method for initializing the pareto set")

    @classmethod
    def all_neighborhood_indices(
        cls, batch_size, objectives_weights, K, distance="cosine"
    ):
        # Calculate the neighborhood of the diverse samples
        neighborhood_indices = []
        for i in range(batch_size):
            neighborhood_indices.append(
                cls.get_neighborhood_indices(
                    objectives_weights[i],
                    objectives_weights,
                    K,
                    distance=distance,
                )
            )
        # shape: (batch_size, K)
        neighborhood_indices = torch.stack(neighborhood_indices, dim=0)
        return neighborhood_indices

    def calculate_scores(
        self, batch_size, t, O, classifiers, batch_diverse_samples, **kwargs
    ):
        """
        classifiers: a list of classifiers, each is a function that takes x
        and returns the score on the i-th objective
        merged_samples_x_1: the samples to predict the scores
        return: the scores for the samples, merged_samples_x_1, shape: (batch_size * O, D)
        """
        # Predict the scores for the diverse samples
        scores = []  # shape: (batch_size, O, len(classifiers))
        # shape: (batch_size * O, D)
        merged_samples = batch_diverse_samples.view(-1, self.D)
        with torch.no_grad():
            # Since the classifiers are trained for x_1, we need to convert the samples
            # to x_1 first
            # shape: (batch_size * O, D)
            time_embedding = self.time_embedding(
                torch.Tensor([t]).to(merged_samples.device)
            ).repeat(batch_size * O, 1)
            merged_samples_u_t = self.vnet(merged_samples, time_embedding)
            merged_samples_x_1 = (
                merged_samples_u_t * (1 - (1 - self.sigma) * t)
                + (1 - self.sigma) * merged_samples
            )
            if "need_repair" in kwargs and kwargs["need_repair"]:
                xl = kwargs["xl"]
                xu = kwargs["xu"]
                merged_samples_x_1 = torch.clip(merged_samples_x_1, xl, xu)
            for classifier in classifiers:
                scores.append(-1 * classifier(merged_samples_x_1))

        # shape: (batch_size * O, len(classifiers))
        scores = torch.stack(scores, dim=1)
        # shape: (batch_size, O, len(classifiers))
        scores = scores.view(batch_size, O, len(classifiers))
        # shape: (batch_size, O, len(classifiers))
        merged_samples_x_1 = merged_samples_x_1.view(batch_size, O, self.D)

        return scores, merged_samples_x_1

    @classmethod
    def get_angle_filter_mask(cls, angles, phi, batch_size):
        """
        angles: the angles between the predicted scores and the i-th objective weights
        phi: the threshold for the angles
        return: the filter mask for the angles
        """
        # shape: (batch_size, K * O, 1)
        angle_filter_mask = angles <= (1 / 2 * phi.unsqueeze(1))
        for i in range(batch_size):
            angle_filter_mask[i, torch.argmin(angles[i])] = True
        # If the angle is smaller than phi_i, we keep the sample, otherwise we set the score to -inf
        return angle_filter_mask

    def repair_boundary(self, solutions, t, xl, xu):
        """
        solutions: the solutions to be repaired  # shape: (batch_size * O, D)
        task: the task to be solved
        xl: the lower bound of the solutions
        xu: the upper bound of the solutions
        convert_to_x1: whether to convert the solutions to x_1 first
        method: the method to repair the solutions
        return: the repaired solutions # shape: (batch_size * O, D)
        """
        # Calculate the lower bound and upper bound at time t
        # shape: (batch_size * O, D), get u_t first
        with torch.no_grad():
            time_embedding = self.time_embedding(
                torch.Tensor([t]).to(solutions.device)
            ).repeat(solutions.shape[0], 1)
            u_t = self.vnet(solutions, time_embedding)

        # Calculate xl and xu at time t
        xl_t = (xl - (1 - (1 - self.sigma) * t) * u_t) / (1 - self.sigma)
        xu_t = (xu - (1 - (1 - self.sigma) * t) * u_t) / (1 - self.sigma)

        # Repair the boundary of the solutions
        # shape: (batch_size * O, D)
        x_t = torch.clip(solutions, xl_t, xu_t)

        return x_t

    @classmethod
    def check_duplicates(cls, pareto_set):
        """
        pareto_set: the pareto set to be checked. Tuple of (solution, score), solution is a tensor, score is a float
        return: the number of duplicates in the pareto set
        """
        # Get all solutions, change dtype to float tensor
        solutions = [solution[0].float() for solution in pareto_set]
        # Check duplicates, if the two tensors are close enough, we consider them as duplicates
        unique_solutions = []
        for solution in solutions:
            if unique_solutions == []:
                unique_solutions.append(solution)
                continue
            else:
                is_duplicate = False
                for unique_solution in unique_solutions:
                    if torch.allclose(solution, unique_solution, atol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_solutions.append(solution)
        return len(solutions) - len(unique_solutions)

    @classmethod
    def remove_duplicates(cls, pareto_set):
        """
        pareto_set: the pareto set to be checked. List of solutions, each is a tensor
        return: the pareto set without duplicates
        """
        pareto_set = [solution.float() for solution in pareto_set]
        unique_solutions = []
        for solution in pareto_set:
            if unique_solutions == []:
                unique_solutions.append(solution)
                continue
            else:
                is_duplicate = False
                for unique_solution in unique_solutions:
                    if torch.allclose(solution, unique_solution, atol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_solutions.append(solution)
        return unique_solutions

    @classmethod
    def get_N_non_dominated_solutions(
        cls, res_x, res_y, N, classifiers, use_true_y=False
    ):
        """
        If use_true_y=True, select using res_y directly.
        This is the safe path for discrete tasks whose classifiers were trained
        on transformed inputs rather than raw decoded inputs.
        """
        if use_true_y:
            fronts = NonDominatedSorting().do(res_y)
            N_best_indices = get_N_nondominated_indices(
                Y=res_y, num_ret=N, fronts=fronts
            )
            return res_x[N_best_indices], res_y[N_best_indices]

        predicted_scores = []
        with torch.no_grad():
            for classifier in classifiers:
                device = next(classifier.parameters()).device
                dtype = next(classifier.parameters()).dtype
                x_t = torch.as_tensor(res_x, device=device, dtype=dtype)
                pred = classifier(x_t)
                predicted_scores.append(pred)

        predicted_res_y = torch.stack(predicted_scores, dim=1).squeeze()
        predicted_res_y = predicted_res_y.detach().cpu().numpy()

        fronts = NonDominatedSorting().do(predicted_res_y)
        N_best_indices = get_N_nondominated_indices(
            Y=predicted_res_y, num_ret=N, fronts=fronts
        )
        return res_x[N_best_indices], res_y[N_best_indices]

    def paretoflow_sample(
        self,
        classifiers,
        T=1000,
        O=10,
        K=20,
        num_solutions=256,
        distance="cosine",
        g_t=0.1,
        init_method="d_best",
        task=None,
        task_name=None,
        t_threshold=0.8,
        adaptive=False,
        gamma=2.0,
    ):
        """
        classifiers: a list of classifiers, each is a function that takes x
        and returns the score on the i-th objective
        T: the number of steps to run the algorithm
        O: the number of offspring to generate at each step
        K: the number of neighborhoods to consider
        num_solutions: the number of solutions to generate
        distance: the distance metric to use
        g_t: the step size for the stochastic Euler method
        init_method: the method to initialize the pareto set
        t_threshold: the threshold for the time step
        adaptive: whether to use the adaptive time step
        gamma: the gamma parameter for the weighted sum of the scores
        """
        _pf_total_t0 = time()

        _pf_device = next(self.parameters()).device
        if _pf_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(_pf_device)

        _pf_sampling_t0 = time()

        # Obtain objectives weights and the number of samples we want to generate
        # shape: (batch_size, len(classifiers))
        objectives_weights, batch_size = FlowMatching.calculate_objectives_weights(
            len(classifiers), num_solutions
        )

        # the pareto set of the generated samples
        # shape: (batch_size, D)
        pareto_set = self.initialize_pareto_set(
            batch_size,
            objectives_weights=objectives_weights,
            task=task,
            methods=init_method,
        )

        # Calculate the neighborhood of the diverse samples
        # shape: (batch_size, K)
        neighborhood_indices = FlowMatching.all_neighborhood_indices(
            batch_size, objectives_weights, K, distance=distance
        )

        # Calculate the M closest indices for filtering the samples in non-convex cases
        # shape: (batch_size, M)
        M = len(classifiers) + 1
        all_m_closest_indices = FlowMatching.all_neighborhood_indices(
            batch_size, objectives_weights, M, distance=distance
        )

        # shape: (batch_size, M, len(classifiers))
        m_closest_objectives_weights = objectives_weights[all_m_closest_indices]
        m_closest_angles = FlowMatching.calculate_angles(
            m_closest_objectives_weights.view(batch_size * M, len(classifiers)),
            objectives_weights.repeat_interleave(
                M, dim=0
            ),  # shape: (batch_size, 1, len(classifiers))
        )
        m_closest_angles = m_closest_angles.view(batch_size, M, 1)
        # shape: (batch_size, 1)
        # #We find the M closest angles but include the angle with itself, so we divide by len(classifiers) to get the average angle
        # because the angle with itself is 0
        phi = 2 * m_closest_angles.sum(dim=1) / len(classifiers)

        # Algorithm 1 in the paper
        # go step-by-step to x_1 (data)
        ts, delta_t = FlowMatching.get_ts_and_delta_t(
            T, t_threshold=t_threshold, adaptive=adaptive
        )

        # Precompute lower bound and upper bound for the repair method
        if (
            task.xl is not None
            and task.xu is not None
            and (
                isinstance(task.problem, DTLZ)
                or isinstance(task.problem, SyntheticProblem)
            )
        ):
            xl = task.xl
            xu = task.xu
            if task.is_discrete:
                xl = task.to_logits(np.int64(xl).reshape(1, -1))
                _, dim, n_classes = tuple(xl.shape)
                xl = xl.reshape(-1, dim * n_classes)
            if task.is_sequence:
                xl = task.to_logits(np.int64(xl).reshape(1, -1))
            # shape: (batch_size * O, D)
            xl = task.normalize_x(xl)
            xl = torch.from_numpy(
                xl
            ).squeeze()  # Add squeeze to remove the first dimension of size 1 when its a discrete task or sequence task
            # shape: (batch_size * O, D)
            xl = xl.unsqueeze(0).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            xl = xl.to(device).type(torch.float32)
            if task.is_discrete:
                xu = task.to_logits(np.int64(xu).reshape(1, -1))
                _, dim, n_classes = tuple(xu.shape)
                xu = xu.reshape(-1, dim * n_classes)
            if task.is_sequence:
                xu = task.to_logits(np.int64(xu).reshape(1, -1))
            # shape: (batch_size * O, D)
            xu = task.normalize_x(xu)
            xu = torch.from_numpy(
                xu
            ).squeeze()  # Add squeeze to remove the first dimension of size 1 when its a discrete task or sequence task
            # shape: (batch_size * O, D)
            xu = xu.unsqueeze(0).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            xu = xu.to(device).type(torch.float32)
            need_repair = True
        else:
            xl = None
            xu = None
            need_repair = False

        # use tqdm for progress bar
        with tqdm(total=T, desc="Conditional Sampling", unit="step") as pbar:
            # sample x_0 first, offspring
            # shape: (batch_size, D)
            x_t = self.sample_base(torch.empty(batch_size, self.D)).to(device)

            # Euler method
            count = 0
            for t in ts[1:]:
                count += 1
                # this is x_t + v(x_t, t, y) * delta_t
                # shape: (batch_size, D)
                x_t = x_t + self.weighted_conditional_vnet(
                    x_t,
                    t - delta_t(t),
                    classifiers,
                    weights=objectives_weights,
                    gamma=gamma,
                    t_threshold=t_threshold,
                    delta_t=delta_t(t),
                ) * delta_t(t)
                if t < t_threshold:
                    if need_repair:
                        x_t = self.repair_boundary(
                            x_t,
                            t,
                            xl[0].unsqueeze(0).repeat(batch_size, 1),
                            xu[0].unsqueeze(0).repeat(batch_size, 1),
                        )
                    pbar.update(1)
                    continue

                # Stochastic Euler method to generate diverse samples
                # shape: (batch_size, O, D)
                batch_diverse_samples = x_t.unsqueeze(1).repeat(1, O, 1)
                # shape: (batch_size, O, D)
                batch_diverse_samples = batch_diverse_samples + g_t * torch.randn_like(
                    batch_diverse_samples
                ) * torch.sqrt(delta_t(t))

                # Repair the boundary of the samples
                if need_repair:
                    batch_diverse_samples = batch_diverse_samples.view(
                        batch_size * O, self.D
                    )
                    batch_diverse_samples = self.repair_boundary(
                        batch_diverse_samples, t, xl, xu
                    )
                    batch_diverse_samples = batch_diverse_samples.view(
                        batch_size, O, self.D
                    )

                    # Calculate the scores for the diverse samples
                    # shape: (batch_size, O, len(classifiers))
                    scores, merged_samples_x_1 = self.calculate_scores(
                        batch_size,
                        t,
                        O,
                        classifiers,
                        batch_diverse_samples,
                        need_repair=True,
                        xl=xl,
                        xu=xu,
                        task=task,
                    )
                else:
                    scores, merged_samples_x_1 = self.calculate_scores(
                        batch_size, t, O, classifiers, batch_diverse_samples, task=task
                    )

                # Calculate the scores for the neighborhood samples
                # Filter the samples to avoid non-convexity as eq 11 in the paper

                # shape: (batch_size, K, O, len(classifiers))
                neighborhood_scores = scores[neighborhood_indices]
                # shape: (batch_size, K * O, len(classifiers))
                neighborhood_scores = neighborhood_scores.view(
                    batch_size, K * O, len(classifiers)
                )

                # Calculate the angles between the predicted scores and the i-th objective weights
                # shape: (batch_size, K * O, 1)
                angles = FlowMatching.calculate_angles(
                    neighborhood_scores.view(batch_size * K * O, len(classifiers)),
                    objectives_weights.repeat_interleave(K * O, dim=0),
                )
                angles = angles.view(batch_size, K * O, 1)

                # Filter out the samples whose angles are larger than phi_i, find the indices
                # shape: (batch_size, K * O, 1)
                angle_filter_mask = FlowMatching.get_angle_filter_mask(
                    angles, phi, batch_size
                )

                # Calculate weighted sum of the scores using the i-th objective weights
                # shape: (batch_size, K * O, len(classifiers))
                weighted_scores = neighborhood_scores * objectives_weights.unsqueeze(
                    1
                ).repeat_interleave(K * O, dim=1)
                # shape: (batch_size, K * O, 1)
                weighted_scores = weighted_scores.sum(dim=-1).unsqueeze(-1)
                weighted_scores[~angle_filter_mask] = float("-inf")
                # Choose the sample with the highest score as the next offspring
                # shape: (batch_size)
                index = torch.argmax(weighted_scores.squeeze(), dim=1)

                # shape: (batch_size, K, O, D)
                neighborhood_designs = batch_diverse_samples[neighborhood_indices]
                # shape: (batch_size, K * O, D)
                neighborhood_designs = neighborhood_designs.view(
                    batch_size, K * O, self.D
                )
                # Use the index to get the next offspring
                # shape: (batch_size, D)
                next_offspring = neighborhood_designs[torch.arange(batch_size), index]

                # Update the pareto set. If the new offspring is better than the i-th solution in the pareto set,
                # replace the i-th solution with the new offspring
                for i in range(batch_size):
                    if weighted_scores[i, index[i]] > pareto_set[i][1]:
                        # shape: (K, O, D)
                        candidates = merged_samples_x_1[neighborhood_indices[i]]
                        # shape: (K * O, D)
                        candidates = candidates.view(K * O, self.D)
                        # shape: (D)
                        pareto_set[i] = (
                            candidates[index[i]].squeeze(),
                            weighted_scores[i, index[i]],
                        )
                # Update x_t
                x_t = next_offspring
                pbar.update(1)

        temp_pareto_set = [pareto_set[i][0] for i in range(batch_size)]
        # Remove duplicates in the pareto set, because they are not contributing to the hypervolume
        temp_pareto_set = FlowMatching.remove_duplicates(temp_pareto_set)
        assert (
            len(temp_pareto_set) >= num_solutions
        ), "Error: The number of solutions in the pareto set is less than the number of solutions we want to keep"
        temp_pareto_set = torch.stack(temp_pareto_set, dim=0).squeeze()

        if _pf_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(_pf_device)

        paretoflow_sampling_seconds = time() - _pf_sampling_t0

        # Final true objective evaluation / HV computation starts here.
        _pf_eval_t0 = time()

        # Check if any of the pareto set elements are out of bounds in the normalized space
        if need_repair:
            oob_ratio = out_of_bounds_ratio_in_normalized_space(
                temp_pareto_set,
                task.normalize_x(task.xl).squeeze(),
                task.normalize_x(task.xu).squeeze(),
            )
        else:
            oob_ratio = 0.0

        temp_pareto_set = task.denormalize_x(temp_pareto_set.cpu().detach().numpy())
        if task.is_discrete:
            temp_pareto_set = task.to_integers(temp_pareto_set)
        nadir_point = task.nadir_point
        # For calculating hypervolume, we use the min-max normalization
        res_x = temp_pareto_set
        res_y = task.predict(res_x)
        # Do a non-dominated sorting to get the pareto set
        res_x, res_y = FlowMatching.get_N_non_dominated_solutions(
            res_x,
            res_y,
            num_solutions,
            classifiers,
            task.is_discrete,
        )
        # Transpose if necessary. This is an issue induced by problematic tasks from the benchmark
        # After communication with the author of the benchmark, we omit to benchmark these tasks
        # To repeat our experiments, we should not transpose the results
        if res_y.shape[0] != res_x.shape[0]:
            res_y = res_y.T
        visible_masks = np.ones(len(res_y))
        visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
        visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
        res_x = res_x[np.where(visible_masks == 1)[0]]
        res_y = res_y[np.where(visible_masks == 1)[0]]

        res_y_75_percent = get_quantile_solutions(res_y, 0.75)
        res_y_50_percent = get_quantile_solutions(res_y, 0.50)
        # To calculate hypervolume, we use the min-max normalization as suggested by the benchmark
        res_y = task.normalize_y(res_y, normalization_method="min-max")
        nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
        res_y_50_percent = task.normalize_y(
            res_y_50_percent, normalization_method="min-max"
        )
        res_y_75_percent = task.normalize_y(
            res_y_75_percent, normalization_method="min-max"
        )

        _, d_best = task.get_N_non_dominated_solutions(
            N=256, return_x=False, return_y=True
        )

        d_best = task.normalize_y(d_best, normalization_method="min-max")

        d_best_hv = hv(nadir_point, d_best, task_name)
        hv_value = hv(nadir_point, res_y, task_name)
        hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
        hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

        delta_spread_100 = spread(res_y)
        delta_spread_75 = spread(res_y_75_percent)
        delta_spread_50 = spread(res_y_50_percent)

        print(f"Hypervolume (100th): {hv_value:4f}")
        print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
        print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
        print(f"Hypervolume (D(best)): {d_best_hv:4f}")

        print(f"Delta-Spread (100th): {delta_spread_100:4f}")
        print(f"Delta-Spread (75th):  {delta_spread_75:4f}")
        print(f"Delta-Spread (50th):  {delta_spread_50:4f}")
        # Save the results
        paretoflow_evaluation_seconds = time() - _pf_eval_t0
        paretoflow_total_wall_seconds = time() - _pf_total_t0

        hv_results = {
            "hypervolume/D(best)": d_best_hv,
            "hypervolume/100th": hv_value,
            "hypervolume/75th": hv_value_75_percentile,
            "hypervolume/50th": hv_value_50_percentile,
            "delta_spread/100th": delta_spread_100,
            "delta_spread/75th": delta_spread_75,
            "delta_spread/50th": delta_spread_50,
            "constraint/out_of_bounds_ratio": oob_ratio,
            "time/paretoflow_sampling_seconds": paretoflow_sampling_seconds,
            "time/paretoflow_evaluation_seconds": paretoflow_evaluation_seconds,
            "time/paretoflow_total_wall_seconds": paretoflow_total_wall_seconds,
            "time/paretoflow_sampling_kind": "iterative_guided_sampling",
        }
        pareto_set = [
            pareto_set[i][0] for i in range(batch_size)
        ]  # shape: (batch_size, D)
        pareto_set = torch.stack(pareto_set, dim=0).squeeze()  # shape: (batch_size, D)
        # convert to numpy array
        pareto_set = pareto_set.cpu().detach().numpy()
        # We return all the pareto set and save them to the file
        # Filter samples later during evaluation if needed
        return pareto_set, hv_results

    def log_prob(self, x_1, reduction="mean"):
        """
        Estimate log p_1(x_1) by integrating the ODE backward from t=1 to t=0
        and accumulating the divergence correction.

        This version is for the newest VectorFieldNet API:

            v = self.vnet(x, t_emb)

        where t_emb is produced by self.time_embedding(t_tensor).

        If you kept the old additive-time architecture, replace:
            self.vnet(x_req, t_emb.detach())
            self.vnet(x_t, t_emb)

        with:
            self.vnet(x_req + t_emb.detach())
            self.vnet(x_t + t_emb)
        """
        was_training = self.training
        self.eval()

        B = x_1.shape[0]
        dev = x_1.device
        dtype = x_1.dtype

        ts = torch.linspace(
            1.0,
            0.0,
            self.T,
            device=dev,
            dtype=dtype,
        )

        x_t = x_1.detach()

        # Approximation of int_0^1 div v_t(x_t) dt.
        div_integral = torch.zeros(B, device=dev, dtype=dtype)

        for i in range(len(ts) - 1):
            t = ts[i]
            dt = ts[i + 1] - ts[i]  # negative because we go 1 -> 0

            t_tensor = torch.full(
                (B, 1),
                float(t.detach().cpu().item()),
                device=dev,
                dtype=dtype,
            )

            t_emb = self.time_embedding(t_tensor)

            with torch.enable_grad():
                x_req = x_t.detach().requires_grad_(True)

                v_req = self.vnet(x_req, t_emb.detach())

                e = torch.randn_like(x_req)

                # Correct Hutchinson estimator:
                #     div v ~= e^T J_v e
                hutchinson_scalar = (v_req * e).sum()

                grad = torch.autograd.grad(
                    hutchinson_scalar,
                    x_req,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]

                div_est = (grad * e).view(B, -1).sum(dim=1)

            # Since dt < 0, forward-time contribution is -dt.
            div_integral = div_integral + div_est.detach() * (-dt)

            with torch.no_grad():
                v_t = self.vnet(x_t, t_emb)
                x_t = x_t + v_t * dt

        log_p_0 = self.log_p_base(x_t, reduction="sum")
        log_p_1 = log_p_0 - div_integral

        if was_training:
            self.train()

        if reduction == "mean":
            return log_p_1.mean()
        elif reduction == "sum":
            return log_p_1.sum()
        elif reduction == "none":
            return log_p_1
        else:
            raise ValueError(f"Unknown reduction={reduction}")
