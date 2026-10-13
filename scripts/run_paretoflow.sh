#!/usr/bin/env bash
# Original ParetoFlow baseline, using its own proxy/flow architecture and settings.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ACTION=${ACTION:-all}
case "$ACTION" in all|sample) ;; *) echo "ACTION must be all or sample" >&2; exit 2;; esac
if [[ ${1:-} == --worker ]]; then
  args=(--task_name="$2" --seed="$3")
  if [[ "$ACTION" == sample ]]; then args+=(--sampling_only=True); fi
  run_command "$PYTHON_BIN" -u experiments/offline_moo/off_moo_baselines/paretoflow/experiment.py "${args[@]}"
  exit
fi
init_scheduler paretoflow
for task in "${TASK_VALUES[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    submit_job "${task}_seed${seed}_${ACTION}" bash "$PROJECT_ROOT/scripts/run_paretoflow.sh" --worker "$task" "$seed"
  done
done
finish_jobs
