import json
import os

import numpy as np
import offline_moo.off_moo_bench as ob
import torch
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.utils import get_quantile_solutions, set_seed
from torch.utils.data import DataLoader

from off_moo_bench.problem.dtlz import DTLZ
from off_moo_bench.problem.synthetic_func import SyntheticProblem

from experiments.paretoflow_args import parse_args
from experiments.paretoflow_nets import FlowMatching, VectorFieldNet
from experiments.paretoflow_utils import (
    ALLTASKSDICT,
    DesignDataset,
    MultipleModels,
    SingleModel,
    SingleModelBaseTrainer,
    get_dataloader,
    tkwargs,
    training,
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from experiments.paretoflow_rfm import (
    resolve_rfm_steering_modes,
    rfm_steered_flow_sample,
)

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def _fmt_float_for_name(x: float) -> str:
    # filesystem-safe float
    s = f"{x:g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def train_proxies(args):
    task_name = ALLTASKSDICT[args.task_name]
    print(f"Task: {task_name}")

    set_seed(args.seed)

    task = ob.make(
        task_name,
        dataset_kwargs=dict(x_normalize_method="z-score", y_normalize_method="z-score"),
    )

    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    n_obj = y.shape[1]
    data_size, n_dim = tuple(X.shape)
    model_save_dir = args.proxies_store_path
    os.makedirs(model_save_dir, exist_ok=True)

    model = MultipleModels(
        n_dim=n_dim,
        n_obj=n_obj,
        train_mode="Vallina",
        hidden_size=[2048, 2048],
        save_dir=args.proxies_store_path,
        save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
    )
    model.set_kwargs(**tkwargs)

    trainer_func = SingleModelBaseTrainer

    for which_obj in range(n_obj):

        y0 = y[:, which_obj].copy().reshape(-1, 1)

        trainer = trainer_func(
            model=list(model.obj2model.values())[which_obj],
            which_obj=which_obj,
            args=args,
        )

        (train_loader, val_loader) = get_dataloader(
            X,
            y0,
            val_ratio=(
                1 - args.proxies_val_ratio
            ),  # means 0.9 for training and 0.1 for validation
            batch_size=args.proxies_batch_size,
        )

        trainer.launch(train_loader, val_loader)


def train_flow_matching(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(
        task_name,
        dataset_kwargs=dict(x_normalize_method="z-score", y_normalize_method="z-score"),
    )
    print(f"Task: {task_name}")

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)

    # For usual cases, we normalize the inputs and outputs with z-score normalization
    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    if args.fm_validation_size is not None:
        total_size = int(X.shape[0])
        val_size = int(args.fm_validation_size)

        if val_size <= 0:
            raise ValueError(
                f"args.fm_validation_size must be positive, got {val_size}"
            )

        if val_size >= total_size:
            raise ValueError(
                f"args.fm_validation_size={val_size} must be smaller than "
                f"dataset size={total_size}"
            )

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(total_size)

        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        X_test = X[val_idx]
        y_test = y[val_idx]

        X = X[train_idx]
        y = y[train_idx]

        print(
            f"Random flow train/val split: "
            f"train_size={X.shape[0]}, val_size={X_test.shape[0]}"
        )
    else:
        raise ValueError(
            "args.fm_validation_size should not be None because "
            "flow validation currently requires a held-out validation set."
        )

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    print(f"Data size: {data_size}")
    print(f"Number of objectives: {n_obj}")
    print(f"Number of dimensions: {n_dim}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Create datasets
    training_dataset = DesignDataset(X)
    val_dataset = DesignDataset(X_test)

    # Create dataloaders
    training_loader = DataLoader(
        training_dataset, batch_size=args.fm_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.fm_batch_size, shuffle=False)

    # Create the model
    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
    )
    model_store_dir = args.fm_store_path
    if not (os.path.exists(model_store_dir)):
        os.makedirs(model_store_dir)

    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps, prob_path=args.fm_prob_path
    )
    model = model.to(device)

    fm_weight_decay = 0.0

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.fm_lr,
        weight_decay=fm_weight_decay,
    )

    print(f"Flow optimizer: AdamW(" f"lr={args.fm_lr}, weight_decay={fm_weight_decay})")

    # Training procedure
    fm_val = training(
        name=model_store_dir + name,
        max_patience=args.fm_patience,
        num_epochs=args.fm_epochs,
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        val_loader=val_loader,
    )

    return fm_val


def sampling(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(
        task_name,
        dataset_kwargs=dict(x_normalize_method="z-score", y_normalize_method="z-score"),
    )
    print(f"Task: {task_name}")

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)

    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    # Obtain the number of objectives
    n_obj = y.shape[1]

    # Use one more neighbor than objectives when K is automatic.
    if args.fm_K == 0:
        fm_K = n_obj + 1
    else:
        fm_K = args.fm_K

    # Obtain the number of data points and the number of dimensions
    data_size, n_dim = tuple(X.shape)

    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + f"_K={args.fm_K}"
        + "_"
        + f"O={args.fm_O}"
        + "_"
        + f"gt={args.fm_gt}"
        + "_"
        + f"gamma={args.fm_gamma}"
    )
    model_name = (
        args.fm_prob_path + "_" + str(1000) + "_" + task_name + "_" + str(args.seed)
    )
    model_store_dir = args.fm_store_path

    # Load the best model
    net = VectorFieldNet(n_dim, args.fm_hidden_size)
    net = net.to(device)
    model_best = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps, prob_path=args.fm_prob_path
    )
    model_best = model_best.to(device)
    model_best = torch.load(model_store_dir + model_name + ".model")
    model_best = model_best.to(device)
    print(
        f"Succesfully loaded the model from {model_store_dir + model_name + '.model'}"
    )

    # Load the classifiers
    list_of_classifiers = []
    for i in range(n_obj):
        classifier = SingleModel(
            input_size=n_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=args.proxies_store_path,
            save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        list_of_classifiers.append(classifier)
    print(f"Loaded {len(list_of_classifiers)} classifiers successfully.")

    # Conditional sampling
    x_samples, hv_results = model_best.paretoflow_sample(
        list_of_classifiers,
        T=args.fm_sampling_steps,
        O=args.fm_O,
        K=fm_K,
        num_solutions=args.fm_num_solutions,
        distance=args.fm_distance_metrics,
        init_method=args.fm_init_method,
        g_t=args.fm_gt,
        task=task,
        task_name=task_name,
        t_threshold=args.fm_threshold,
        adaptive=args.fm_adaptive,
        gamma=args.fm_gamma,
    )

    # Denormalize the solutions
    res_x = x_samples
    if args.normalization:
        res_x = task.denormalize_x(res_x)

    if task.is_discrete:
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)
    if res_y.shape[0] != res_x.shape[0]:
        res_y = res_y.T

    # Store the results
    if not (os.path.exists(args.samples_store_path)):
        os.makedirs(args.samples_store_path)

    np.save(args.samples_store_path + name + "_x.npy", res_x)
    np.save(args.samples_store_path + name + "_y.npy", res_y)

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)

    return res_x, res_y


def evaluation(args):
    # Set the seed
    set_seed(args.seed)

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(task_name)
    print(f"Task: {task_name}")

    # Get the data
    name = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
        + f"_K={args.fm_K}"
        + "_"
        + f"O={args.fm_O}"
        + "_"
        + f"gt={args.fm_gt}"
        + "_"
        + f"gamma={args.fm_gamma}"
    )
    res_x = np.load(args.samples_store_path + name + "_x.npy")
    res_y = np.load(args.samples_store_path + name + "_y.npy")
    print(f"Loaded the generated samples from {args.samples_store_path}")

    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    # For calculating hypervolume, we use the min-max normalization
    res_y = task.normalize_y(res_y, normalization_method="min-max")
    nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
    res_y_50_percent = task.normalize_y(
        res_y_50_percent, normalization_method="min-max"
    )
    res_y_75_percent = task.normalize_y(
        res_y_75_percent, normalization_method="min-max"
    )

    _, d_best = task.get_N_non_dominated_solutions(N=256, return_x=False, return_y=True)

    d_best = task.normalize_y(d_best, normalization_method="min-max")

    d_best_hv = hv(nadir_point, d_best, task_name)
    hv_value = hv(nadir_point, res_y, task_name)
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")

    # Save the results
    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
    }

    if not (os.path.exists(args.results_store_path)):
        os.makedirs(args.results_store_path)

    with open(args.results_store_path + name + "_hv_results.json", "w") as f:
        json.dump(hv_results, f, indent=4)


def sampling_rfm(args):
    # Set the seed
    set_seed(args.seed)

    stage1_steering_mode, apg_stage2_projector_mode = resolve_rfm_steering_modes(
        sampling_variant=args.rfm_sampling_variant,
        stage1_steering_mode=args.rfm_stage1_steering_mode,
        apg_stage2_projector_mode=args.rfm_apg_stage2_projector_mode,
    )

    # Get the task
    task_name = ALLTASKSDICT[args.task_name]
    task = ob.make(
        task_name,
        dataset_kwargs=dict(x_normalize_method="z-score", y_normalize_method="z-score"),
    )
    print(f"Task: {task_name}")

    # Get the data
    X = task.x.copy()
    y = task.y.copy()

    if task.is_discrete:
        X = task.to_logits(X)

    if args.normalization:
        X = task.normalize_x(X)
        y = task.normalize_y(y)

    # number of objectives
    n_obj = y.shape[1]

    # dims
    _, n_dim = tuple(X.shape)

    # --------------------------
    # Load Flow model (same as sampling(args))
    # --------------------------
    model_name = (
        args.fm_prob_path + "_" + str(1000) + "_" + task_name + "_" + str(args.seed)
    )
    model_store_dir = args.fm_store_path

    net = VectorFieldNet(n_dim, args.fm_hidden_size).to(device)
    model_best = FlowMatching(
        net, args.fm_sigma, n_dim, args.fm_sampling_steps, prob_path=args.fm_prob_path
    ).to(device)

    model_best = torch.load(model_store_dir + model_name + ".model").to(device)
    print(
        f"Succesfully loaded the model from {model_store_dir + model_name + '.model'}"
    )

    # --------------------------
    # Load classifiers (same as sampling(args))
    # --------------------------
    list_of_classifiers = []
    for i in range(n_obj):
        classifier = SingleModel(
            input_size=n_dim,
            which_obj=i,
            hidden_size=[2048, 2048],
            save_dir=args.proxies_store_path,
            save_prefix=f"MultipleModels-Vallina-{task_name}-{args.seed}",
        )
        classifier.load()
        classifier = classifier.to(device)
        classifier.eval()
        list_of_classifiers.append(classifier)
    print(f"Loaded {len(list_of_classifiers)} classifiers successfully.")

    # ==========================================================
    # RFM-steered unconditional flow sampling
    # ==========================================================
    name_tag_rfm = (
        args.fm_prob_path
        + "_"
        + str(args.fm_sampling_steps)
        + "_"
        + task_name
        + "_"
        + str(args.seed)
    )

    strengths_tag = "-".join(_fmt_float_for_name(s) for s in args.rfm_strengths)

    is_morl_task = args.task_name in {
        "mo_hopper_v2",
        "mo_swimmer_v2",
    }

    morl_low_rank_tag = ""
    if is_morl_task:
        morl_low_rank_tag = f"_lrk{int(args.rfm_low_rank_k)}"

    # ----------------------------------------------------------
    # RFM filename tag. Keep method identity explicit and compact.
    # ----------------------------------------------------------
    rfm_tag = (
        "rfm"
        + f"_s{strengths_tag}"
        + f"_repair{int(args.rfm_repair)}"
        + f"_{args.rfm_sampling_variant}"
        + "_improved_proxy"
    )

    if args.rfm_sampling_variant == "adaptive_noise":
        if stage1_steering_mode == "adaptive_hard":
            rfm_tag += "_stage1AdaptiveHard"
        else:
            rfm_tag += (
                "_softStage1" + f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
            )
        rfm_tag += "_decodeOnly"

    elif args.rfm_sampling_variant == "adaptive_projected_guidance":
        if stage1_steering_mode == "adaptive_hard":
            rfm_tag += "_stage1AdaptiveHard"
        else:
            rfm_tag += (
                "_softStage1" + f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
            )

        if apg_stage2_projector_mode == "adaptive_hard":
            rfm_tag += "_stage2AdaptiveHard"
        else:
            rfm_tag += (
                "_stage2SoftSpectral"
                + f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
            )

        rfm_tag += (
            f"_t{_fmt_float_for_name(args.rfm_guidance_start_t)}"
            + f"_alpha{_fmt_float_for_name(args.rfm_guidance_alpha_max)}"
        )

    elif args.rfm_sampling_variant == "data_adaptive_manifold_guidance":
        if stage1_steering_mode == "adaptive_hard":
            rfm_tag += "_stage1AdaptiveHard"
        else:
            rfm_tag += "_softStage1"

        rfm_tag += (
            "_endpointGrad"
            + "_fixedLocalKnnFastR1Logged"
            + ("_softOnlineFilter" if stage1_steering_mode == "adaptive_hard" else "")
            + f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
            + f"_t{_fmt_float_for_name(args.rfm_guidance_start_t)}"
            + f"_ttan{_fmt_float_for_name(args.rfm_tangent_start_t)}"
            + f"_tau{_fmt_float_for_name(args.rfm_tangent_tau)}"
            + f"_knn{int(args.rfm_tangent_k_nn)}"
            + f"_tr{max(1, int(args.rfm_tangent_refresh_every))}"
            + f"_kr{max(1, int(args.rfm_knn_refresh_every))}"
            + f"_nmbeta{_fmt_float_for_name(args.rfm_norm_match_beta)}"
            + f"_aclip{_fmt_float_for_name(args.rfm_guidance_alpha_max)}"
        )

    elif args.rfm_sampling_variant == "crd_reduced":
        if stage1_steering_mode == "adaptive_hard":
            rfm_tag += "_stage1AdaptiveHard"
        else:
            rfm_tag += "_softStage1"

        rfm_tag += (
            "_endpointGrad"
            + "_normalizedJ"
            + (
                f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
                if stage1_steering_mode == "soft_spectral"
                else ""
            )
            + f"_t{_fmt_float_for_name(args.rfm_guidance_start_t)}"
            + f"_nmbeta{_fmt_float_for_name(args.rfm_norm_match_beta)}"
            + f"_gatec{_fmt_float_for_name(args.rfm_robust_gate_c)}"
        )

    elif args.rfm_sampling_variant == "cone_projected":
        if stage1_steering_mode == "adaptive_hard":
            rfm_tag += "_stage1AdaptiveHard"
        else:
            rfm_tag += "_softStage1"

        rfm_tag += (
            "_endpointGrad"
            + "_normalizedJ"
            + (
                f"_gamma{_fmt_float_for_name(args.rfm_soft_rfm_gamma)}"
                if stage1_steering_mode == "soft_spectral"
                else ""
            )
            + f"_t{_fmt_float_for_name(args.rfm_guidance_start_t)}"
            + f"_nmbeta{_fmt_float_for_name(args.rfm_norm_match_beta)}"
        )

    else:
        raise ValueError(f"Unknown rfm_sampling_variant={args.rfm_sampling_variant!r}")

    rfm_tag += morl_low_rank_tag

    name_rfm = name_tag_rfm + "_" + rfm_tag

    # Optional bounds for final hard clipping (in normalized-x space)
    xl = xu = None
    if (
        args.rfm_repair
        and (task.xl is not None)
        and (task.xu is not None)
        and (
            isinstance(task.problem, DTLZ) or isinstance(task.problem, SyntheticProblem)
        )
    ):
        xl_np = np.asarray(task.xl).reshape(1, -1)
        xu_np = np.asarray(task.xu).reshape(1, -1)

        if task.is_discrete:
            xl_np = task.to_logits(np.int64(xl_np))
            xu_np = task.to_logits(np.int64(xu_np))

        if args.normalization:
            xl_np = task.normalize_x(xl_np)
            xu_np = task.normalize_x(xu_np)

        xl = xl_np.squeeze()
        xu = xu_np.squeeze()
        need_repair = True
    else:
        need_repair = False

    # Run RFM steered sampling
    rfm_out = rfm_steered_flow_sample(
        flow=model_best,
        classifiers=list_of_classifiers,
        task=task,
        task_name=task_name,
        cache_task_key=args.task_name,
        rfm_seed=args.seed,
        cache_dir=args.rfm_cache_dir,
        retrain_rfm=args.retrain_rfm,
        retrain_rfm_stage2=args.retrain_rfm_stage2,
        stage1_steering_mode=stage1_steering_mode,
        apg_stage2_projector_mode=apg_stage2_projector_mode,
        n_rfm_train=args.rfm_n_train,
        n_gen_per_obj=args.rfm_n_gen_per_obj,
        strengths=tuple(args.rfm_strengths),
        weight_mode=args.rfm_weight_mode,
        n_weights=args.rfm_n_weights,
        metric_weight_mode=args.rfm_metric_weight_mode,
        metric_n_weights=args.rfm_metric_n_weights,
        dirichlet_alpha=args.rfm_dirichlet_alpha,
        dd_partitions=args.rfm_dd_partitions,
        L=args.rfm_L,
        lam=args.rfm_lam,
        T=args.rfm_T,
        num_solutions=args.rfm_num_solutions,
        xl=xl,
        xu=xu,
        repair=need_repair,
        t_threshold=args.rfm_t_threshold,
        normalize_M_trace=args.rfm_normalize_M_trace,
        metric_jitter=args.rfm_metric_jitter,
        sampling_variant=args.rfm_sampling_variant,
        guidance_start_t=args.rfm_guidance_start_t,
        guidance_alpha_max=args.rfm_guidance_alpha_max,
        norm_match_beta=args.rfm_norm_match_beta,
        soft_rfm_gamma=args.rfm_soft_rfm_gamma,
        tangent_start_t=args.rfm_tangent_start_t,
        tangent_tau=args.rfm_tangent_tau,
        tangent_k_nn=args.rfm_tangent_k_nn,
        tangent_refresh_every=args.rfm_tangent_refresh_every,
        knn_refresh_every=args.rfm_knn_refresh_every,
        tangent_verbose_log=args.rfm_tangent_verbose_log,
        robust_gate_c=args.rfm_robust_gate_c,
        low_rank_k=args.rfm_low_rank_k,
        low_rank_niter=args.rfm_low_rank_niter,
    )

    hv_results_rfm = rfm_out["hv_results"]
    res_x_rfm = rfm_out["X_denorm"]
    res_y_rfm = rfm_out["Y_raw"]

    hv_results_rfm["rfm/file_tag"] = str(rfm_tag)

    # ==========================================================
    # Save RFM samples
    # ==========================================================
    os.makedirs(args.samples_store_path, exist_ok=True)
    os.makedirs(args.results_store_path, exist_ok=True)
    np.save(args.samples_store_path + name_rfm + "_x.npy", res_x_rfm)
    np.save(args.samples_store_path + name_rfm + "_y.npy", res_y_rfm)

    # ==========================================================
    # Save RFM HV metrics, including overall wall-clock fields.
    # ==========================================================
    with open(args.results_store_path + name_rfm + "_hv_results.json", "w") as f:
        json.dump(hv_results_rfm, f, indent=4)

    return res_x_rfm, res_y_rfm
