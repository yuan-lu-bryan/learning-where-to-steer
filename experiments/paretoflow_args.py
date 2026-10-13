"""This module contains the function to parse command line arguments."""

from argparse import ArgumentParser, ArgumentTypeError

from experiments.paretoflow_utils import all_task_names


def str2bool(v):
    """
    Convert a string to a boolean value. This function is used as a type for the ArgumentParser
    to avoid potential errors when parsing boolean values.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected.")


def parse_args():  # Parse command line arguments
    """
    Parse the command line arguments for the ParetoFlow
    """
    parser = ArgumentParser(
        description="Noise-space steering and flow-guided offline multi-objective optimization"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="The random seed for the experiment"
    )
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        choices=all_task_names,
        help="Which task we will use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train_proxies",
            "train_flow_matching",
            "sampling",
            "sampling_rfm",
            "evaluation",
        ],
        help="Which experiment mode to run.",
    )
    parser.add_argument(
        "--normalization",
        type=str2bool,
        nargs="?",
        default=True,
        help="True denotes normalizing the inputs and outputs",
    )
    parser.add_argument(
        "--fm_adaptive",
        type=str2bool,
        nargs="?",
        default=False,
        help="True denotes using adaptive time step in the flow matching model",
    )
    parser.add_argument(
        "--fm_validation_size",
        default=1000,
        type=int,
        help="Size of the validation set for training the flow matching model",
    )
    parser.add_argument(
        "--fm_batch_size", default=128, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--fm_prob_path", default="icfm", type=str, help="The path to the probability"
    )
    parser.add_argument(
        "--fm_hidden_size",
        default=512,
        type=int,
        help="The number of neurons in scale (s) and translation (t) nets",
    )
    parser.add_argument(
        "--fm_sigma", default=0.0, type=float, help="Sigma used in flow matching"
    )
    parser.add_argument(
        "--fm_gamma", default=2.0, type=float, help="Gamma used to control the guidance"
    )
    parser.add_argument(
        "--fm_sampling_steps",
        default=1000,
        type=int,
        help="The number of sampling steps, i.e., T",
    )
    parser.add_argument(
        "--fm_epochs",
        default=1000,
        type=int,
        help="Number of epochs to train the flow matching model",
    )
    parser.add_argument(
        "--fm_lr",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer for training the flow matching model",
    )
    parser.add_argument(
        "--fm_patience",
        default=20,
        type=int,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--fm_O",
        default=5,
        type=int,
        help="The number of stochastic samples we generated for each objective weights",
    )
    parser.add_argument(
        "--fm_K",
        default=0,
        type=int,
        help="The number of neighbors we used for each objective weights",
    )
    parser.add_argument(
        "--fm_num_solutions",
        default=256,
        type=int,
        help="The number of solutions we keep by using non-dominated sorting",
    )
    parser.add_argument(
        "--fm_gt",
        default=0.1,
        type=float,
        help="The coefficient for the random noise in the stochastic euler method",
    )
    parser.add_argument(
        "--fm_threshold",
        default=0.8,
        type=float,
        help="When time > threshold, we use our algorithm to generate samples",
    )
    parser.add_argument(
        "--fm_distance_metrics",
        default="cosine",
        choices=["cosine", "euclidean"],
        type=str,
        help="The metrtics used to calculate the distance between objectives weights",
    )
    parser.add_argument(
        "--fm_init_method",
        default="d_best",
        choices=["empty_init", "d_best"],
        type=str,
        help="empty_init: initialize the PS with empty solutions, d_best: initialize the PS with the best solutions in the offline dataset",
    )
    parser.add_argument(
        "--fm_store_path",
        default="flow_matching_models/",
        type=str,
        help="The folder to store the trained flow matching model",
    )
    parser.add_argument(
        "--samples_store_path",
        default="generated_samples/",
        type=str,
        help="The folder to store the generated samples",
    )
    parser.add_argument(
        "--results_store_path",
        default="saved_metrics/",
        type=str,
        help="The folder to store the computed metrics in hyper-volume",
    )
    parser.add_argument(
        "--config_store_path",
        default="log_configs/",
        type=str,
        help="The folder to store the log configurations",
    )
    parser.add_argument(
        "--proxies_store_path",
        default="proxies_model/",
        type=str,
        help="The folder to store the trained proxies model",
    )
    parser.add_argument(
        "--proxies_epochs",
        default=200,
        type=int,
        help="Number of epochs to train the proxies model",
    )
    parser.add_argument(
        "--proxies_lr",
        default=1e-3,
        type=float,
        help="Learning rate for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_lr_decay",
        default=0.98,
        type=float,
        help="Learning rate decay for the optimizer for training proxies model",
    )
    parser.add_argument(
        "--proxies_batch_size",
        default=128,
        type=int,
        help="Batch size for training the proxies model",
    )
    parser.add_argument(
        "--proxies_val_ratio",
        default=0.1,
        type=float,
        help="The ratio of the validation set",
    )
    # -------------------------
    # RFM sampling args
    # -------------------------
    parser.add_argument(
        "--rfm_n_train",
        default=10000,
        type=int,
        help="Number of (noise, objective) pairs to build the RFM dataset.",
    )
    parser.add_argument(
        "--rfm_n_gen_per_obj",
        default=512,
        type=int,
        help="Number of generated samples per WEIGHT per steering strength (name kept for compatibility).",
    )
    parser.add_argument(
        "--rfm_strengths",
        nargs="+",
        type=float,
        default=[10.0],
        help="Steering strengths s used in z0 + s*v. Example: --rfm_strengths 2 5 10",
    )
    parser.add_argument(
        "--rfm_L",
        default=10.0,
        type=float,
        help="Mahalanobis Laplace kernel bandwidth L in RFM.",
    )
    parser.add_argument(
        "--rfm_lam",
        default=1e-3,
        type=float,
        help="Kernel ridge regression regularization lambda in RFM.",
    )
    parser.add_argument(
        "--rfm_T",
        default=5,
        type=int,
        help="Number of RFM iterations (AGOP recursion steps).",
    )
    parser.add_argument(
        "--rfm_num_solutions",
        default=256,
        type=int,
        help="Number of solutions retained for HV computation in RFM sampling.",
    )
    parser.add_argument(
        "--rfm_normalize_M_trace",
        type=str2bool,
        nargs="?",
        default=False,
        help="Optional stabilization: normalize trace(M) each RFM iteration (not in paper).",
    )
    parser.add_argument(
        "--rfm_repair",
        type=str2bool,
        nargs="?",
        default=True,
        help="Whether to use repair_boundary during unconditional flow sampling in RFM.",
    )
    parser.add_argument(
        "--rfm_t_threshold",
        default=0.8,
        type=float,
        help="Only apply repair_boundary when t >= rfm_t_threshold.",
    )

    # -------------------------
    # RFM weights / shared-metric recursion args
    # -------------------------
    parser.add_argument(
        "--rfm_weight_mode",
        type=str,
        default="dirichlet",
        choices=["dirichlet", "das-dennis", "per_obj"],
        help="Weight distribution used for shared-metric RFM and steering vectors.",
    )
    parser.add_argument(
        "--rfm_n_weights",
        default=64,
        type=int,
        help="Number of weights used to build C=E[ww^T] and number of steering directions.",
    )
    parser.add_argument(
        "--rfm_dirichlet_alpha",
        default=1.0,
        type=float,
        help="Dirichlet concentration alpha (only if rfm_weight_mode=dirichlet).",
    )
    parser.add_argument(
        "--rfm_dd_partitions",
        default=0,
        type=int,
        help="Das–Dennis partitions (0 => auto choose to reach rfm_n_weights).",
    )

    parser.add_argument(
        "--rfm_metric_jitter",
        default=1e-6,
        type=float,
        help="Jitter used inside metric factorization and kernel distance computation.",
    )

    parser.add_argument(
        "--rfm_low_rank_k",
        default=32,
        type=int,
        help=(
            "Low-rank basis dimension q used by torch.svd_lowrank(M_w) for "
            "high-dimensional MORL-style tasks. Only active in RFM basis "
            "extractors when D > 2000."
        ),
    )

    parser.add_argument(
        "--rfm_low_rank_niter",
        default=2,
        type=int,
        help=(
            "Number of subspace iterations for torch.svd_lowrank(M_w). "
            "Only active in RFM basis extractors when D > 2000."
        ),
    )
    # -------------------------
    # RFM sampling variant
    # -------------------------
    parser.add_argument(
        "--rfm_sampling_variant",
        type=str,
        default="adaptive_projected_guidance",
        choices=[
            "adaptive_noise",
            "adaptive_projected_guidance",
            "data_adaptive_manifold_guidance",
            "crd_reduced",
            "cone_projected",
        ],
        help=(
            "RFM sampling variant. "
            "All five variants support adaptive-hard or soft-spectral Stage-1 "
            "noise steering. adaptive_projected_guidance couples that choice "
            "to its Stage-2 projected guidance. data_adaptive_manifold_guidance "
            "keeps its soft-RFM online gradient filter and cached state-space "
            "kNN-local Gram-PCA tangent projection in either Stage-1 mode. "
            "crd_reduced and cone_projected retain their normalized-J online "
            "guidance rules without online RFM filtering or tangent probes."
        ),
    )

    parser.add_argument(
        "--rfm_guidance_start_t",
        default=0.7,
        type=float,
        help=(
            "Guidance start time. For adaptive_projected_guidance, this is also "
            "the Stage-2 RFM construction time. For data-adaptive and both CRD "
            "variants, this is when endpoint-gradient guidance turns on."
        ),
    )

    parser.add_argument(
        "--rfm_guidance_alpha_max",
        default=8.0,
        type=float,
        help=(
            "Guidance strength cap. For adaptive_projected_guidance, this is the "
            "final value of the linear ramp. For data_adaptive_manifold_guidance, "
            "this is alpha_clip. The two CRD variants intentionally do not clip "
            "their norm-matched guidance."
        ),
    )

    parser.add_argument(
        "--rfm_norm_match_beta",
        default=0.25,
        type=float,
        help=(
            "Norm-matched guidance fraction for data-adaptive, CRD-Reduced, "
            "and cone-projected online guidance. Only data-adaptive applies "
            "the guidance_alpha_max clip."
        ),
    )

    parser.add_argument(
        "--rfm_stage1_steering_mode",
        type=str,
        default=None,
        choices=["adaptive_hard", "soft_spectral"],
        help=(
            "Stage-1 noise-steering family for any of the five RFM variants. "
            "If omitted, reproduce the repository's legacy behavior: "
            "adaptive_hard for adaptive_noise/adaptive_projected_guidance and "
            "soft_spectral for data_adaptive_manifold_guidance/crd_reduced/"
            "cone_projected. APG applies the selected family to both stages."
        ),
    )

    parser.add_argument(
        "--rfm_apg_stage2_projector_mode",
        type=str,
        default=None,
        choices=["adaptive_hard", "soft_spectral"],
        help=(
            "Deprecated compatibility alias for APG Stage-2. If supplied, it "
            "selects both APG stages when --rfm_stage1_steering_mode is omitted, "
            "or must equal that option when both are supplied. Mixed hard/soft "
            "APG stages are rejected. Prefer the Stage-1 flag."
        ),
    )

    parser.add_argument(
        "--rfm_soft_rfm_gamma",
        default=1.0,
        type=float,
        help=(
            "Spectral sharpness gamma for full- or low-rank soft RFM weights: "
            "rho_j=(lambda_j/lambda_1)^gamma. Used by every selected "
            "soft-spectral stage and by DAMG's established online soft-RFM "
            "gradient filter."
        ),
    )

    parser.add_argument(
        "--rfm_tangent_start_t",
        default=0.8,
        type=float,
        help=(
            "kNN-local tangent-projection onset for " "data_adaptive_manifold_guidance."
        ),
    )

    parser.add_argument(
        "--rfm_tangent_tau",
        default=0.2,
        type=float,
        help=(
            "Relative singular-value floor tau for each local-neighborhood "
            "Gram-PCA rank estimate. The shared rank is the median local rank; "
            "must be in (0,1)."
        ),
    )

    parser.add_argument(
        "--rfm_tangent_k_nn",
        default=12,
        type=int,
        help=(
            "Number of state-space neighbors used by the fixed-localspace "
            "DAMG tangent estimator. Must be in [2, batch_size-1]."
        ),
    )

    parser.add_argument(
        "--rfm_tangent_refresh_every",
        default=1,
        type=int,
        help=(
            "Refresh interval, in ODE steps, for the cached local Gram-PCA "
            "tangent bases. Default 1 is fixed_localspace_fast_r1."
        ),
    )

    parser.add_argument(
        "--rfm_knn_refresh_every",
        default=15,
        type=int,
        help=(
            "Refresh interval, in ODE steps, for the state-space kNN graph "
            "used by data_adaptive_manifold_guidance."
        ),
    )

    parser.add_argument(
        "--rfm_tangent_verbose_log",
        type=str2bool,
        nargs="?",
        default=True,
        help=(
            "Print one fixed-localspace tangent diagnostic summary per "
            "data_adaptive_manifold_guidance sampling call. The same "
            "diagnostics are saved in the HV JSON regardless of this flag."
        ),
    )

    parser.add_argument(
        "--rfm_robust_gate_c",
        default=0.1,
        type=float,
        help=(
            "CRD-Reduced gate scale c in "
            "omega=||d_cons||^2/(||d_cons||^2+c). Ignored by other variants."
        ),
    )

    # -------------------------
    # RFM Cache Agrs
    # -------------------------
    parser.add_argument(
        "--rfm_cache_dir",
        default="experiments/results/rfm_cache_adaptive_variants_v3",
        type=str,
        help=(
            "Directory used to cache all offline RFM/AGOP Stage-1 and "
            "Stage-2 objects. Defaults to the shared cache directory used "
            "by scripts/sample_rfm.sh."
        ),
    )

    parser.add_argument(
        "--retrain_rfm",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "If True, recompute and overwrite every required cached RFM/AGOP "
            "stage. "
            "If False, load cached RFM/AGOP objects when available."
        ),
    )

    parser.add_argument(
        "--retrain_rfm_stage2",
        type=str2bool,
        nargs="?",
        default=False,
        help=(
            "If True, rebuild only adaptive_projected_guidance's Stage-2 RFM "
            "cache for the selected Stage-1 steering mode. A matching Stage-1 "
            "cache is still loaded and reused. Switching the coupled APG "
            "steering family does not require this flag because hard and soft "
            "Stage-2 caches use distinct filenames. Ignored by variants "
            "without a Stage-2 cache."
        ),
    )

    parser.add_argument(
        "--rfm_metric_weight_mode",
        type=str,
        default="same",
        choices=["same", "dirichlet", "das-dennis", "per_obj"],
        help=(
            "Weight mode used only for the shared RFM metric recursion. "
            "'same' means use the same weight tensor as sampling-time weights."
        ),
    )

    parser.add_argument(
        "--rfm_metric_n_weights",
        default=0,
        type=int,
        help=(
            "Number of weights used only for the shared RFM metric recursion. "
            "0 means use rfm_n_weights."
        ),
    )
    # -------------------------
    # Plotting args
    # -------------------------
    parser.add_argument(
        "--plots_store_path",
        default="plots/",
        type=str,
        help="Directory to save comparison plots. If None, uses results_store_path + 'plots/'.",
    )
    parser.add_argument(
        "--pf_gt_n_points",
        default=4000,
        type=int,
        help="Number of points used to plot the ground-truth PF (when available, e.g., ZDT).",
    )
    # -------------------------
    # SIRSTEER / surrogate-free Pareto steering args
    # -------------------------
    parser.add_argument(
        "--sir_method",
        type=str,
        default="global_sir",
        choices=["global_sir", "local_sir", "unsteered"],
        help="SIRSTEER method: global_sir, local_sir, or unsteered.",
    )
    parser.add_argument(
        "--sir_beta",
        default=10.0,
        type=float,
        help="Steering magnitude beta in z_steered = z + beta * v.",
    )
    parser.add_argument(
        "--sir_n_pref",
        default=256,
        type=int,
        help="Number of preference vectors.",
    )
    parser.add_argument(
        "--sir_n_per_pref",
        default=2,
        type=int,
        help="Number of generated samples per preference vector.",
    )
    parser.add_argument(
        "--sir_weight_mode",
        type=str,
        default="das-dennis",
        choices=["das-dennis", "dirichlet", "linspace"],
        help=(
            "Preference-vector generation for SIRSTEER. "
            "'das-dennis' gives deterministic simplex reference directions; "
            "'dirichlet' gives random simplex weights; "
            "'linspace' is only for 2-objective tasks."
        ),
    )
    parser.add_argument(
        "--sir_dd_partitions",
        default=0,
        type=int,
        help=(
            "Das-Dennis partition count. 0 means choose the smallest partition "
            "count that gives at least sir_n_pref directions."
        ),
    )
    parser.add_argument(
        "--sir_n_slices",
        default=15,
        type=int,
        help="Number of SIR slices.",
    )
    parser.add_argument(
        "--sir_k_neighbors",
        default=200,
        type=int,
        help="Number of nearest neighbors for local_sir.",
    )
    parser.add_argument(
        "--sir_n_steps",
        default=0,
        type=int,
        help="Euler steps inside SteerableFlow. 0 means use fm_sampling_steps.",
    )
    parser.add_argument(
        "--sir_eval_num_solutions",
        default=256,
        type=int,
        help="Number of nondominated solutions retained for HV and delta-spread computation.",
    )

    args = parser.parse_args()
    return args
