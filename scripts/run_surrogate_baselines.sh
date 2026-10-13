#!/usr/bin/env bash
# Eleven paper DNN baselines, each optimized with NSGA-II on learned objectives.
# COM is retained as an optional MultipleModels training mode outside Table 1.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODEL=${MODEL:-all}
case "$MODEL" in all) MODEL_VALUES=(end2end multi_head multiple_models);;
  end2end|multi_head|multiple_models) MODEL_VALUES=("$MODEL");;
  *) echo "MODEL must be all, end2end, multi_head, or multiple_models" >&2; exit 2;;
esac
if [[ "$MODEL" == all && -n "${TRAIN_MODES:-}" ]]; then
  echo "Select one MODEL when overriding TRAIN_MODES." >&2; exit 2
fi
set_model_modes() {
  local item allowed defaults mode_input
  case "$1" in
    end2end) MODEL_NAME=End2End; allowed='Vallina GradNorm PcGrad'; defaults=$allowed;;
    multi_head) MODEL_NAME=MultiHead; allowed='Vallina GradNorm PcGrad'; defaults=$allowed;;
    multiple_models) MODEL_NAME=MultipleModels; allowed='Vallina IOM RoMA ICT TriMentoring COM'; defaults='Vallina IOM RoMA ICT TriMentoring';;
  esac
  mode_input=${TRAIN_MODES:-$defaults}
  read -r -a MODE_VALUES <<< "${mode_input//$'\n'/ }"
  validate_list TRAIN_MODES "${MODE_VALUES[@]}"
  for item in "${MODE_VALUES[@]}"; do
    [[ " $allowed " == *" $item "* ]] || { echo "Unsupported $1 training mode: $item" >&2; exit 2; }
  done
}
# Validate requested modes before launching any work.
for model in "${MODEL_VALUES[@]}"; do set_model_modes "$model"; done
if [[ ${1:-} == --worker ]]; then
  model=$2 task=$3 seed=$4 mode=$5
  set_model_modes "$model"
  args=(--model="$MODEL_NAME" --train_mode="$mode" --task="$task"
    --use_wandb=False --retrain_model=False --seed="$seed")
  if [[ "$model" == end2end ]]; then args+=(--solver_n_gen=200); fi
  run_command "$PYTHON_BIN" -u "experiments/offline_moo/off_moo_baselines/$model/experiment.py" "${args[@]}"
  exit
fi
init_scheduler surrogate_baselines
for model in "${MODEL_VALUES[@]}"; do
  set_model_modes "$model"
  for task in "${TASK_VALUES[@]}"; do
    for seed in "${SEED_VALUES[@]}"; do
      for mode in "${MODE_VALUES[@]}"; do
        submit_job "${model}_${mode}_${task}_seed${seed}" bash "$PROJECT_ROOT/scripts/run_surrogate_baselines.sh" --worker "$model" "$task" "$seed" "$mode"
      done
    done
  done
done
finish_jobs
