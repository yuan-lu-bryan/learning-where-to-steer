"""
pareto_plotting.py
==================
Raw-J score-space plotting utilities for Combined Robust diagnostics.

Convention:
  - Raw objectives f are minimization objectives.
  - The sampler logs scores s_hat = -f_hat, so larger is better.
  - ``y_hat`` is treated only as a backward-compatible alias for ``s_hat``.
  - Local dot-product diagnostics use raw ``J_hat`` by default, not
    ``J_hat_normalized``.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def pareto_front_mask(Y: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Return a boolean mask of non-dominated rows of Y (shape (N,m))."""
    Y = np.asarray(Y, dtype=np.float64)
    if not maximize:
        Y = -Y
    N = Y.shape[0]
    nd = np.ones(N, dtype=bool)
    for i in range(N):
        if not nd[i]:
            continue
        dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        dominated[i] = False
        if dominated.any():
            nd[i] = False
    return nd


def _get_score(info: Dict) -> np.ndarray:
    return np.asarray(info.get("s_hat", info.get("y_hat")), dtype=np.float64)


def _get_raw_J(info: Dict) -> np.ndarray:
    if "J_hat" not in info:
        raise KeyError("info_log entry has no raw J_hat field")
    return np.asarray(info["J_hat"], dtype=np.float64)


def compute_consensus_diagnostics(
    info_log: List[Dict],
    w: np.ndarray,
    primary_w_index: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract score-space quantities from an info_log.

    Uses raw J_hat for local directional derivatives:

        JT_d_cons  = J_hat^T d_cons
        JT_d_align = J_hat^T d_align
        JT_g       = J_hat^T g_tilde or g_robust

    All score quantities use s_hat = -f_hat. Larger is better.
    """
    w = np.asarray(w, dtype=np.float64)

    ts, ss, Jdc, Jda, Jg, oms, ws = [], [], [], [], [], [], []
    gate_cs, dcsq, alpha, relg = [], [], [], []
    scalar_pre, scalar_post, ret_post = [], [], []

    for info in info_log:
        J = _get_raw_J(info)
        s = _get_score(info)
        pw = info["per_w"][primary_w_index]

        d_cons = np.asarray(pw["d_cons"], dtype=np.float64)
        d_align = np.asarray(pw.get("d_align", pw.get("d_align_raw")), dtype=np.float64)
        g = np.asarray(info.get("g_tilde", info.get("g_robust")), dtype=np.float64)

        ts.append(float(info["t"]))
        ss.append(s)
        Jdc.append(J.T @ d_cons)
        Jda.append(J.T @ d_align)
        Jg.append(J.T @ g)
        oms.append(float(pw.get("omega", np.nan)))
        ws.append(float(np.dot(w, s)))

        gate_cs.append(float(info.get("gate_c", np.nan)))
        dcsq.append(float(pw.get("d_cons_sq", np.dot(d_cons, d_cons))))
        alpha.append(float(info.get("alpha", np.nan)))
        relg.append(float(info.get("relative_guidance_norm", np.nan)))
        scalar_pre.append(
            float(
                info.get(
                    "scalar_effect_pre_filter",
                    info.get("scalar_effect_g_robust", np.nan),
                )
            )
        )
        scalar_post.append(float(info.get("scalar_effect_post_filter", np.nan)))
        ret_post.append(float(info.get("effect_retention_post_filter", np.nan)))

    s_arr = np.stack(ss)
    Jdc_arr = np.stack(Jdc)
    Jda_arr = np.stack(Jda)
    Jg_arr = np.stack(Jg)

    return {
        "t": np.asarray(ts),
        "s_hat": s_arr,
        "y_hat": s_arr,
        "JT_d_cons": Jdc_arr,
        "JT_d_align": Jda_arr,
        "JT_g": Jg_arr,
        "omega": np.asarray(oms),
        "w_dot_s": np.asarray(ws),
        "w_dot_y": np.asarray(ws),
        "gate_c": np.asarray(gate_cs),
        "d_cons_sq": np.asarray(dcsq),
        "alpha": np.asarray(alpha),
        "relative_guidance_norm": np.asarray(relg),
        "scalar_effect_pre_filter": np.asarray(scalar_pre),
        "scalar_effect_post_filter": np.asarray(scalar_post),
        "effect_retention_post_filter": np.asarray(ret_post),
    }


def plot_consensus_diagnostics(
    info_log: List[Dict],
    w: np.ndarray,
    save_path: str = "consensus_diagnostics.png",
):
    """Four-panel raw-J consensus/alignment/effect diagnostics."""
    import matplotlib.pyplot as plt

    diags = compute_consensus_diagnostics(info_log, w)
    t = diags["t"]
    s = diags["s_hat"]
    Jdc = diags["JT_d_cons"]
    Jda = diags["JT_d_align"]
    Jg = diags["JT_g"]
    om = diags["omega"]
    ws = diags["w_dot_s"]
    ret = diags["effect_retention_post_filter"]
    m = s.shape[1]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for i in range(m):
        ax[0, 0].plot(t, s[:, i], label=rf"$s_{i+1}$")
    ax[0, 0].set_title(r"(a) Score trajectories $s=-\hat f$ at $\hat{x}_1$")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].legend()

    for i in range(m):
        ax[0, 1].plot(t, Jdc[:, i], label=rf"$g_{i+1}\cdot d_{{cons}}$")
    ax[0, 1].axhline(0.0, ls="--", alpha=0.6)
    n_viol = int((Jdc < -1e-9).sum())
    ax[0, 1].set_title(
        f"(b) Raw-J consensus effects $J^T d_{{cons}}$\nnegative entries: {n_viol}/{Jdc.size}"
    )
    ax[0, 1].set_xlabel("t")
    ax[0, 1].legend()

    for i in range(m):
        ax[1, 0].plot(t, Jg[:, i], label=rf"$g_{i+1}\cdot g$")
    ax[1, 0].axhline(0.0, ls="--", alpha=0.6)
    n_viol_g = int((Jg < -1e-9).sum())
    ax[1, 0].set_title(
        f"(c) Final guided raw effects $J^T g$\nnegative entries: {n_viol_g}/{Jg.size}"
    )
    ax[1, 0].set_xlabel("t")
    ax[1, 0].legend()

    ax[1, 1].plot(t, om, label=r"$\omega$")
    ax[1, 1].plot(
        t, ws / (np.nanmax(np.abs(ws)) + 1e-12), ls="--", label=r"$w^T s$ normalized"
    )
    if not np.all(np.isnan(ret)):
        ax[1, 1].plot(t, ret, ls=":", label="effect retention")
    ax[1, 1].axhline(0.0, ls="--", alpha=0.3)
    ax[1, 1].axhline(1.0, ls="--", alpha=0.3)
    ax[1, 1].set_title("(d) Gate, scalar score, and retention")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].legend()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_trajectory(
    info_log: List[Dict],
    save_path: str = "objective_trajectory.png",
    offline_scores: Optional[np.ndarray] = None,
    pareto_scores: Optional[np.ndarray] = None,
    Y_data: Optional[np.ndarray] = None,
    info_log_baseline: Optional[List[Dict]] = None,
    maximize: bool = True,
    label_guided: str = "guided",
):
    """
    Plot the score trajectory in 2D or first two score coordinates.

    Backward-compatible with the current paretoflow.py call, which passes:
        Y_data=...
        info_log_baseline=...
        maximize=True
        label_guided=...
    """
    import matplotlib.pyplot as plt

    scores = np.stack([_get_score(info) for info in info_log])

    if scores.shape[1] < 2:
        raise ValueError("Need at least two objectives/scores to plot trajectory.")

    # Current paretoflow.py passes Y_data already in score orientation.
    if offline_scores is None and Y_data is not None:
        offline_scores = Y_data

    plt.figure(figsize=(7, 6))

    if offline_scores is not None:
        offline_scores = np.asarray(offline_scores, dtype=np.float64)
        if offline_scores.ndim == 2 and offline_scores.shape[1] >= 2:
            plt.scatter(
                offline_scores[:, 0],
                offline_scores[:, 1],
                s=8,
                alpha=0.25,
                label="offline scores" if maximize else "offline values",
            )

    if pareto_scores is not None:
        pareto_scores = np.asarray(pareto_scores, dtype=np.float64)
        if pareto_scores.ndim == 2 and pareto_scores.shape[1] >= 2:
            plt.scatter(
                pareto_scores[:, 0],
                pareto_scores[:, 1],
                s=18,
                alpha=0.55,
                label="offline PF scores",
            )

    if info_log_baseline is not None and len(info_log_baseline) > 0:
        base_scores = np.stack([_get_score(info) for info in info_log_baseline])
        if base_scores.shape[1] >= 2:
            plt.plot(
                base_scores[:, 0],
                base_scores[:, 1],
                marker=".",
                ms=2,
                alpha=0.65,
                label="baseline trajectory",
            )

    plt.plot(
        scores[:, 0],
        scores[:, 1],
        marker="o",
        ms=3,
        label=label_guided,
    )
    plt.scatter(scores[0, 0], scores[0, 1], s=60, marker="s", label="start")
    plt.scatter(scores[-1, 0], scores[-1, 1], s=80, marker="*", label="end")

    plt.xlabel(r"$s_1=-\hat f_1$")
    plt.ylabel(r"$s_2=-\hat f_2$")
    plt.title("Score-space trajectory at denoised endpoint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pareto_scan(
    final_scores: np.ndarray,
    target_weights: Optional[np.ndarray] = None,
    offline_scores: Optional[np.ndarray] = None,
    save_path: str = "pareto_scan.png",
):
    """Scatter final score values from many target weights."""
    import matplotlib.pyplot as plt

    final_scores = np.asarray(final_scores, dtype=np.float64)
    if final_scores.ndim != 2 or final_scores.shape[1] < 2:
        raise ValueError("final_scores must be (N,m) with m>=2")

    plt.figure(figsize=(7, 6))
    if offline_scores is not None:
        offline_scores = np.asarray(offline_scores, dtype=np.float64)
        mask = pareto_front_mask(offline_scores, maximize=True)
        plt.scatter(
            offline_scores[:, 0],
            offline_scores[:, 1],
            s=8,
            alpha=0.2,
            label="offline scores",
        )
        plt.scatter(
            offline_scores[mask, 0],
            offline_scores[mask, 1],
            s=18,
            alpha=0.6,
            label="offline score PF",
        )

    if target_weights is not None and target_weights.shape[1] >= 2:
        c = target_weights[:, 0]
        plt.scatter(
            final_scores[:, 0],
            final_scores[:, 1],
            s=30,
            c=c,
            label="generated final scores",
        )
        plt.colorbar(label="target w[0]")
    else:
        plt.scatter(
            final_scores[:, 0], final_scores[:, 1], s=30, label="generated final scores"
        )

    plt.xlabel(r"$s_1=-\hat f_1$")
    plt.ylabel(r"$s_2=-\hat f_2$")
    plt.title("Final score scan across target weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def summarize_info_log(info_log: List[Dict], w: np.ndarray) -> Dict[str, float]:
    """Return lightweight plotting-side summary statistics."""
    d = compute_consensus_diagnostics(info_log, w)
    out = {
        "n_steps": int(len(info_log)),
        "median_omega": _nanmedian(d["omega"]),
        "median_gate_c": _nanmedian(d["gate_c"]),
        "median_scalar_effect_pre_filter": _nanmedian(d["scalar_effect_pre_filter"]),
        "median_scalar_effect_post_filter": _nanmedian(d["scalar_effect_post_filter"]),
        "median_effect_retention_post_filter": _nanmedian(
            d["effect_retention_post_filter"]
        ),
        "median_relative_guidance_norm": _nanmedian(d["relative_guidance_norm"]),
        "frac_negative_final_effect_entries": float(np.mean(d["JT_g"] < 0.0)),
    }
    return out


def _nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    return None if np.all(np.isnan(x)) else float(np.nanmedian(x))
