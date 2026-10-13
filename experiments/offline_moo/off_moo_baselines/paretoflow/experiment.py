import datetime
import json
import os
import sys
from time import time

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_PATH)

from off_moo_baselines.paretoflow.paretoflow_args import parse_args
from off_moo_baselines.paretoflow.paretoflow_experiments import (
    evaluation,
    sampling,
    train_flow_matching,
    train_proxies,
)


def run(config: dict):
    args = parse_args()

    # ---- Structured paths (no timestamp dependency) ----
    logging_dir = config["logging_dir"]
    model_save_dir = config["model_save_dir"]

    sample_store_path = os.path.join(logging_dir, "generated_samples")
    results_store_path = os.path.join(logging_dir, "saved_metrics")
    config_store_path = os.path.join(logging_dir, "log_configs")

    proxies_store_path = os.path.join(model_save_dir, "paretoflow_proxy_models")
    fm_store_path = os.path.join(model_save_dir, "flow_matching_models")

    os.makedirs(sample_store_path, exist_ok=True)
    os.makedirs(results_store_path, exist_ok=True)
    os.makedirs(config_store_path, exist_ok=True)
    os.makedirs(proxies_store_path, exist_ok=True)
    os.makedirs(fm_store_path, exist_ok=True)

    # IMPORTANT: baseline config uses key "task", but paretoflow_args uses "task_name"
    task = config.get("task", config.get("task_name", "re21"))

    args.__dict__.update(
        {
            "task_name": task,
            "seed": config["seed"],
            "samples_store_path": sample_store_path,
            "results_store_path": results_store_path,
            "config_store_path": config_store_path,
            "proxies_store_path": proxies_store_path,
            "fm_store_path": fm_store_path,
        }
    )

    name = f"{task}_{config['seed']}"
    config_dict = vars(args)
    with open(os.path.join(args.config_store_path, f"{name}.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    start_time = time()

    if args.sampling_only:
        print("Skipping training, performing sampling only...")
        sampling(args)
    else:
        train_proxies(args)
        train_flow_matching(args)
        sampling(args)
    print(f"Total time: {time() - start_time} seconds")


if __name__ == "__main__":
    from utils import process_args

    # ---- shim: make --task_name drive process_args task loading ----
    orig_argv = sys.argv[:]
    task_from_cli = None
    has_task = False
    for a in orig_argv[1:]:
        if a.startswith("--task_name="):
            task_from_cli = a.split("=", 1)[1]
        if a.startswith("--task="):
            has_task = True

    # If user passed --task_name but not --task, inject --task for process_args ONLY
    argv_for_cfg = orig_argv[:]
    if task_from_cli is not None and not has_task:
        argv_for_cfg.append(f"--task={task_from_cli}")

    sys.argv = argv_for_cfg
    config = process_args(return_dict=True)
    sys.argv = orig_argv

    # Ensure the runtime uses the CLI task_name (even if default.yaml had a task)
    if task_from_cli is not None:
        config["task"] = task_from_cli

    # FORCE stable roots under experiments/offline_moo (ignore default.yaml ./results ./model)
    RESULTS_ROOT = os.path.join(BASE_PATH, "results", "paretoflow")
    MODEL_ROOT = os.path.join(BASE_PATH, "model", "paretoflow")

    config["results_dir"] = RESULTS_ROOT
    config["model_save_dir"] = MODEL_ROOT

    task = config.get("task", "re21")
    seed = config.get("seed", 0)

    run_name = f"ParetoFlow-seed{seed}-{task}"
    logging_dir = config.get(
        "logging_dir", os.path.join(config["results_dir"], run_name)
    )
    os.makedirs(logging_dir, exist_ok=True)
    config["logging_dir"] = logging_dir

    os.makedirs(config["model_save_dir"], exist_ok=True)

    run(config)
