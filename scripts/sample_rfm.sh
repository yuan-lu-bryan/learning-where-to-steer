#!/usr/bin/env bash
# Main RFM methods. METHODS=paretoflow_plus uses the same improved flow/proxies.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# auto preserves the primary repository launcher's method-specific defaults.
STEERING_MODE=${RFM_STAGE1_STEERING_MODE:-auto}
case "$STEERING_MODE" in auto|adaptive_hard|soft_spectral) ;; *) echo "Invalid RFM_STAGE1_STEERING_MODE" >&2; exit 2;; esac
method_input=${METHODS:-adaptive_noise adaptive_projected_guidance data_adaptive_manifold_guidance crd_reduced cone_projected}
read -r -a METHOD_VALUES <<< "${method_input//$'\n'/ }"
validate_list METHODS "${METHOD_VALUES[@]}"
for method in "${METHOD_VALUES[@]}"; do
  case "$method" in adaptive_noise|adaptive_projected_guidance|data_adaptive_manifold_guidance|crd_reduced|cone_projected|paretoflow_plus) ;;
    *) echo "Unknown sampling method: $method" >&2; exit 2;; esac
done
if [[ ${1:-} == --worker ]]; then
  task=$2 seed=$3 method=$4
  job_steering_mode=$STEERING_MODE
  if [[ "$job_steering_mode" == auto ]]; then
    case "$method" in
      adaptive_noise|adaptive_projected_guidance) job_steering_mode=adaptive_hard;;
      *) job_steering_mode=soft_spectral;;
    esac
  fi
  FLOW_ARGS=(--fm_gamma=10.0 --fm_num_solutions=256 --fm_sampling_steps=1000
    --fm_hidden_size=512 --fm_sigma=0.0 --fm_prob_path=icfm)
  if [[ "$method" == paretoflow_plus ]]; then
    run_command "$PYTHON_BIN" -u experiments/paretoflow.py --mode=sampling \
      --task_name="$task" --seed="$seed" "${PATH_ARGS[@]}" "${FLOW_ARGS[@]}"
  else
    n_train=10000 rfm_steps=7 rank=32 rank_iters=2 start_t=0.7
    if is_morl "$task"; then n_train=5000 rfm_steps=4 rank=100 rank_iters=3; fi
    if [[ "$method" == adaptive_projected_guidance ]]; then start_t=0.8; fi
    run_command "$PYTHON_BIN" -u experiments/paretoflow.py --mode=sampling_rfm \
      --task_name="$task" --seed="$seed" "${PATH_ARGS[@]}" "${FLOW_ARGS[@]}" \
      --fm_lr=2e-4 --fm_epochs=1000 --rfm_sampling_variant="$method" \
      --rfm_stage1_steering_mode="$job_steering_mode" --rfm_num_solutions=256 \
      --rfm_n_train="$n_train" --rfm_n_gen_per_obj=2 --rfm_T="$rfm_steps" \
      --rfm_L=20.0 --rfm_lam=1e-3 --rfm_strengths 10.0 \
      --rfm_weight_mode=das-dennis --rfm_n_weights=256 \
      --rfm_metric_weight_mode=same --rfm_metric_n_weights=1024 \
      --rfm_low_rank_k="$rank" --rfm_low_rank_niter="$rank_iters" \
      --rfm_cache_dir="${RFM_CACHE_DIR:-experiments/results/rfm_cache_adaptive_variants_v3}" \
      --retrain_rfm="${RETRAIN_RFM:-False}" --retrain_rfm_stage2="${RETRAIN_RFM_STAGE2:-False}" \
      --rfm_guidance_start_t="$start_t" --rfm_guidance_alpha_max=11.0 --rfm_norm_match_beta=0.5 \
      --rfm_soft_rfm_gamma=0.3 --rfm_tangent_start_t=0.7 --rfm_tangent_tau=0.2 \
      --rfm_tangent_k_nn=12 --rfm_tangent_refresh_every=1 --rfm_knn_refresh_every=15 \
      --rfm_tangent_verbose_log=True --rfm_robust_gate_c=0.1
  fi
  exit
fi
init_scheduler sample_rfm
for task in "${TASK_VALUES[@]}"; do
  for seed in "${SEED_VALUES[@]}"; do
    for method in "${METHOD_VALUES[@]}"; do
      submit_job "${task}_seed${seed}_${method}_${STEERING_MODE}" bash "$PROJECT_ROOT/scripts/sample_rfm.sh" --worker "$task" "$seed" "$method"
    done
  done
done
finish_jobs
