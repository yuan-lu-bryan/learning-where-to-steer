#!/usr/bin/env bash
# Optional GP-based baselines provided by Offline-MOO; not Table 1 methods.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode_input=${TRAIN_MODES:-Vallina ParEGO JES}
read -r -a MODE_VALUES <<< "${mode_input//$'\n'/ }"
validate_list TRAIN_MODES "${MODE_VALUES[@]}"
for mode in "${MODE_VALUES[@]}"; do
  case "$mode" in Vallina|ParEGO|JES) ;; *) echo "Unknown MOBO mode: $mode" >&2; exit 2;; esac
done
if [[ ${1:-} == --worker ]]; then
  run_command "$PYTHON_BIN" -u experiments/offline_moo/off_moo_baselines/mobo/experiment.py \
    --model=MOBO --train_mode="$4" --task="$2" --use_wandb=False --retrain_model=False --seed="$3"
  exit
fi
init_scheduler mobo
for task in "${TASK_VALUES[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    for mode in "${MODE_VALUES[@]}"; do
      submit_job "${mode}_${task}_seed${seed}" bash "$PROJECT_ROOT/scripts/run_mobo.sh" --worker "$task" "$seed" "$mode"
    done
  done
done
finish_jobs
