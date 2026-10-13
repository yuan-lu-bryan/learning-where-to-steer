#!/usr/bin/env bash
# Train the shared proxies and flow used by RFM and ParetoFlow+.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ACTION=${ACTION:-all}
case "$ACTION" in all|proxies|flow) ;; *) echo "ACTION must be all, proxies, or flow" >&2; exit 2;; esac
if [[ ${1:-} == --worker ]]; then
  task=$2 seed=$3
  if [[ "$ACTION" == all || "$ACTION" == proxies ]]; then
    run_command "$PYTHON_BIN" -u experiments/paretoflow.py --task_name="$task" --seed="$seed" \
      --mode=train_proxies --proxies_store_path="$PROXIES_STORE" \
      --config_store_path="experiments/results/log_configs_${RUN_TAG}/" \
      --proxies_epochs=5000 --proxies_lr=1e-3 --proxies_lr_decay=0.995 \
      --proxies_batch_size=128 --proxies_val_ratio=0.1
  fi
  if [[ "$ACTION" == all || "$ACTION" == flow ]]; then
    epochs=1000; if is_morl "$task"; then epochs=3000; fi
    run_command "$PYTHON_BIN" -u experiments/paretoflow.py --task_name="$task" --seed="$seed" \
      --mode=train_flow_matching "${PATH_ARGS[@]}" --fm_patience=20 --fm_lr=2e-4 \
      --fm_gamma=2.0 --fm_epochs="$epochs" --fm_batch_size=128 --fm_validation_size=1000 \
      --fm_sampling_steps=1000 --fm_hidden_size=512 --fm_sigma=0.0 --fm_prob_path=icfm
  fi
  exit
fi
init_scheduler train_rfm
for task in "${TASK_VALUES[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    submit_job "${task}_seed${seed}_${ACTION}" bash "$PROJECT_ROOT/scripts/train_rfm.sh" --worker "$task" "$seed"
  done
done
finish_jobs
