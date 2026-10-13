#!/usr/bin/env bash
# Pareto-conditioned diffusion. Reuses an existing final checkpoint when present.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ACTION=${ACTION:-all}
case "$ACTION" in all|sample) ;; *) echo "ACTION must be all or sample" >&2; exit 2;; esac
if [[ ${1:-} == --worker ]]; then
  task=$2 seed=$3 domain=$(task_domain "$2")
  save_root="$PROJECT_ROOT/experiments/offline_moo/results/pcd"
  args=(--task_name "$task" --seed "$seed" --domain "$domain"
    --sampling-method reference-direction --sampling-guidance-scale 2.5
    --sampling-ref-dir-method energy --sampling-noise-scale 0.05 --num-cond-points 32
    --experiment_name reweight-ref-dir --save_dir "$save_root"
    --data_preserved_ratio 0.2 --reweight-loss)
  if [[ "$ACTION" == sample ]]; then
    args+=(--pcd-model-load-dir "$save_root/$domain/$task/reweight-ref-dir/$seed")
  fi
  # Run beside PCD's local model modules; gin files resolve from their module path.
  cd "$PROJECT_ROOT/experiments/offline_moo/off_moo_baselines/pcd"
  run_command "$PYTHON_BIN" -u train.py "${args[@]}"
  exit
fi
init_scheduler pcd
for task in "${TASK_VALUES[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    submit_job "${task}_seed${seed}_${ACTION}" bash "$PROJECT_ROOT/scripts/run_pcd.sh" --worker "$task" "$seed"
  done
done
finish_jobs
