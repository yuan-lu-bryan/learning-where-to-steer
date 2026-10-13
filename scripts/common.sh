#!/usr/bin/env bash
# Shared configuration and one-job-per-GPU scheduling for the public launchers.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
THREADS=${THREADS:-8}
DRY_RUN=${DRY_RUN:-0}
RUN_TAG=${RUN_TAG:-enhanced_training_v1}
export PYTHONPATH="$PROJECT_ROOT/experiments/offline_moo:$PROJECT_ROOT:$PROJECT_ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export WANDB_DISABLED=true
cd "$PROJECT_ROOT"

validate_list() {
  local name=$1 item seen=' '
  shift
  (( $# > 0 )) || { echo "$name must contain at least one value" >&2; exit 2; }
  for item in "$@"; do
    [[ "$seen" != *" $item "* ]] || { echo "Duplicate $name value: $item" >&2; exit 2; }
    seen+="$item "
  done
}

NON_MORL_TASKS="re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61
 dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7 zdt1 zdt2 zdt3 zdt4 zdt6
 c10mop1 c10mop2 c10mop3 c10mop4 c10mop5 c10mop6 c10mop7 c10mop8 c10mop9
 in1kmop1 in1kmop2 in1kmop3 in1kmop4 in1kmop5 in1kmop6 in1kmop7 in1kmop8 in1kmop9"
MORL_TASKS="mo_hopper_v2 mo_swimmer_v2"
ALL_TASKS="$NON_MORL_TASKS $MORL_TASKS"
# Flatten whitespace before reading arrays; task identifiers contain no spaces.
ALL_TASKS=${ALL_TASKS//$'\n'/ }
NON_MORL_TASKS=${NON_MORL_TASKS//$'\n'/ }
case ${TASK_GROUP:-all} in
  all) DEFAULT_TASKS=$ALL_TASKS ;;
  non_morl) DEFAULT_TASKS=$NON_MORL_TASKS ;;
  morl) DEFAULT_TASKS=$MORL_TASKS ;;
  *) echo "TASK_GROUP must be all, non_morl, or morl" >&2; exit 2 ;;
esac
task_input=${TASKS:-$DEFAULT_TASKS}
seed_input=${SEEDS:-0 1 2 3 4}
read -r -a TASK_VALUES <<< "${task_input//$'\n'/ }"
read -r -a SEED_VALUES <<< "${seed_input//$'\n'/ }"
validate_list TASKS "${TASK_VALUES[@]}"
validate_list SEEDS "${SEED_VALUES[@]}"
for task in "${TASK_VALUES[@]}"; do
  [[ " $ALL_TASKS " == *" $task "* ]] || { echo "Unsupported paper task: $task" >&2; exit 2; }
done
for seed in "${SEED_VALUES[@]}"; do
  [[ "$seed" =~ ^(0|[1-9][0-9]*)$ && ${#seed} -le 10 ]] && (( seed < 4294967296 )) || {
    echo "Seed must be an integer from 0 to 4294967295 without leading zeros: $seed" >&2; exit 2;
  }
done
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "THREADS must be positive" >&2; exit 2; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

FM_STORE="experiments/results/flow_matching_models_${RUN_TAG}/"
PROXIES_STORE="experiments/results/proxies_model_${RUN_TAG}/"
PATH_ARGS=(--fm_store_path="$FM_STORE" --proxies_store_path="$PROXIES_STORE"
  --samples_store_path="experiments/results/generated_samples_${RUN_TAG}/"
  --results_store_path="experiments/results/saved_metrics_${RUN_TAG}/"
  --config_store_path="experiments/results/log_configs_${RUN_TAG}/"
  --plots_store_path="experiments/results/plots_${RUN_TAG}/")

is_morl() { [[ "$1" == mo_hopper_v2 || "$1" == mo_swimmer_v2 ]]; }
task_domain() {
  case "$1" in re*) echo re;; dtlz*|zdt*) echo synthetic;; c10mop*|in1kmop*) echo monas;; mo_*) echo morl;; esac
}
print_command() { printf '%q ' "$@"; printf '\n'; }
run_command() {
  print_command "$@"
  if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}

# Initialization is called by the parent launcher, never by its internal worker.
init_scheduler() {
  local name=$1 gpu lock_fd gpu_input
  LOG_DIR=${LOG_DIR:-"$PROJECT_ROOT/experiments/logs/public_release/$name"}
  if [[ "$DRY_RUN" == 0 && -z "${GPUS:-}" ]]; then
    echo "Set GPUS to the GPU IDs allocated to this run (for example GPUS='0 1')." >&2
    exit 2
  fi
  gpu_input=${GPUS:-dry-run}
  read -r -a GPU_VALUES <<< "${gpu_input//$'\n'/ }"
  validate_list GPUS "${GPU_VALUES[@]}"
  declare -gA GPU_PID=()
  FAILED=0
  if [[ "$DRY_RUN" == 0 ]]; then
    command -v setsid >/dev/null
    command -v flock >/dev/null
    mkdir -p "$LOG_DIR" "$PROJECT_ROOT/experiments/logs/public_release/gpu_locks"
    # paretoflow.py creates its config directory without exist_ok; create the
    # shared roots before concurrent workers can race on first-time setup.
    mkdir -p "$FM_STORE" "$PROXIES_STORE" \
      "experiments/results/generated_samples_${RUN_TAG}" \
      "experiments/results/saved_metrics_${RUN_TAG}" \
      "experiments/results/log_configs_${RUN_TAG}" "experiments/results/plots_${RUN_TAG}"
    for gpu in "${GPU_VALUES[@]}"; do
      [[ "$gpu" =~ ^(0|[1-9][0-9]*)$ || "$gpu" =~ ^GPU-[a-zA-Z0-9-]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
      [[ ! -v "GPU_PID[$gpu]" ]] || { echo "Duplicate GPU ID: $gpu" >&2; exit 2; }
      GPU_PID[$gpu]=''
      exec {lock_fd}>"$PROJECT_ROOT/experiments/logs/public_release/gpu_locks/$gpu.lock"
      flock -n "$lock_fd" || { echo "Another public launcher holds GPU $gpu" >&2; exit 1; }
    done
    trap 'stop_workers; exit 130' INT
    trap 'stop_workers; exit 143' TERM
  fi
}
stop_workers() {
  local gpu pid
  for gpu in "${GPU_VALUES[@]}"; do
    pid=${GPU_PID[$gpu]:-}
    if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  for gpu in "${GPU_VALUES[@]}"; do
    pid=${GPU_PID[$gpu]:-}
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
}
reap_worker() {
  local gpu=$1 pid=${GPU_PID[$1]:-} code
  if [[ -n "$pid" ]]; then
    if wait "$pid"; then code=0; else code=$?; FAILED=1; fi
    printf '[EXIT] GPU %s PID %s status=%s\n' "$gpu" "$pid" "$code"
    GPU_PID[$gpu]=''
  fi
}
submit_job() {
  local label=$1 gpu pid log
  shift
  if [[ "$DRY_RUN" == 1 ]]; then
    # Execute only the launcher's command builder; run_command prints its argv.
    DRY_RUN=1 "$@"
    return
  fi
  while true; do
    for gpu in "${GPU_VALUES[@]}"; do
      pid=${GPU_PID[$gpu]:-}
      if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then reap_worker "$gpu"; pid=''; fi
      if [[ -z "$pid" ]]; then
        log="$LOG_DIR/$label.log"
        printf '[LAUNCH] GPU %s %s -> %s\n' "$gpu" "$label" "$log"
        setsid env CUDA_VISIBLE_DEVICES="$gpu" "$@" >>"$log" 2>&1 &
        GPU_PID[$gpu]=$!
        return
      fi
    done
    sleep 0.2
  done
}
finish_jobs() {
  local gpu
  [[ "$DRY_RUN" == 1 ]] && return
  for gpu in "${GPU_VALUES[@]}"; do reap_worker "$gpu"; done
  return "$FAILED"
}
