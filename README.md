# Learning Where to Steer

**Learning Where to Steer: Noise-Space Geometry for Efficient Offline Multi-Objective Optimization with Generative Models**

Yuan Lu\*, Esha Singh\*, Yi-An Ma, Yusu Wang  
UC San Diego · \*Equal contribution

[Paper](https://arxiv.org/abs/2609.38920) · [PDF](https://arxiv.org/pdf/2609.38920)

Active-Subspace Noise Steering (ASNS) uses Recursive Feature Machines to learn useful directions in the initial noise of a trained flow model. Steering this noise lets us optimize scalarized objectives while keeping the ODE sampler fixed. This repository includes ASNS, its guidance variants, and the baselines used in the paper.

## Installation

Use Linux x86_64 (glibc 2.28 or later), Conda, and an NVIDIA GPU. The paper used Python 3.8, PyTorch 2.0.1, CUDA 11.8, and RTX A6000 GPUs with 48 GB memory. Run all commands from the repository root:

```bash
conda env create -f environment.yml
conda activate ctrl-moo
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.23.2 Cython==0.29.37
python -m pip install -r requirements.txt
python -m pip check
```

The launchers set the local import paths; no `pip install -e .` step is needed. Use the pinned PyTorch 2.0.1 / CUDA 11.8 environment and dependencies, particularly `pymoo==0.5.0`, for the benchmark experiments. The PyTorch command follows the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v201).

Download the [Off-MOO-Bench datasets](https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT) and extract them under `experiments/offline_moo/data/`, preserving the task folders and filenames. These offline samples are required for all tasks, including RE, DTLZ, and ZDT. Checkpoints are produced by training.

The core installation covers the 27 RE, DTLZ, and ZDT tasks. Running all 47 tasks also requires the NAS and MORL dependencies and assets in [Task backends](#task-backends).

## Quickstart

Train the shared flow and objective predictors, then run the five RFM variants on ZDT2 with one seed:

```bash
GPUS="0" TASKS="zdt2" SEEDS="0" ACTION=all bash scripts/train_rfm.sh
GPUS="0" TASKS="zdt2" SEEDS="0" RFM_STAGE1_STEERING_MODE=soft_spectral \
  bash scripts/sample_rfm.sh
```

To inspect commands without starting a job:

```bash
DRY_RUN=1 TASKS="zdt2" SEEDS="0" bash scripts/train_rfm.sh
DRY_RUN=1 TASKS="zdt2" SEEDS="0" RFM_STAGE1_STEERING_MODE=soft_spectral \
  bash scripts/sample_rfm.sh
```

## RFM methods

| Method | `METHODS` identifier |
| --- | --- |
| RFM-AN / ASNS | `adaptive_noise` |
| RFM-APG | `adaptive_projected_guidance` |
| RFM-DAMG | `data_adaptive_manifold_guidance` |
| RFM-CRD-Reduced | `crd_reduced` |
| RFM-Cone | `cone_projected` |

The paper compares AN, APG, DAMG, and Cone against 14 baselines. CRD-Reduced is an additional variant available here. All five share the same trained flow and predictors.

After completing both NAS and MORL backend setup, run the full benchmark with the four RFM methods and soft steering used in the paper (Appendix A.3, Table 4). Leave `TASKS` unset; the commands below select all 47 tasks and seeds `0 1 2 3 4`:

```bash
GPUS="0 1 2 3" THREADS=8 TASK_GROUP=all SEEDS="0 1 2 3 4" ACTION=all \
  bash scripts/train_rfm.sh
GPUS="0 1 2 3" THREADS=8 TASK_GROUP=all SEEDS="0 1 2 3 4" \
  RFM_STAGE1_STEERING_MODE=soft_spectral \
  METHODS="adaptive_noise adaptive_projected_guidance data_adaptive_manifold_guidance cone_projected" \
  bash scripts/sample_rfm.sh
```

Omit `METHODS` to also run CRD-Reduced. The default task group is `all` and the default seeds are `0 1 2 3 4`; `TASKS`, `TASK_GROUP`, and `SEEDS` can narrow the run.

Use `ACTION=proxies` or `ACTION=flow` to train one component. Use `METHODS` to select samplers:

```bash
GPUS="0" TASKS="zdt2" SEEDS="0" \
  RFM_STAGE1_STEERING_MODE=soft_spectral METHODS="adaptive_noise cone_projected" \
  bash scripts/sample_rfm.sh
```

The examples above explicitly select `soft_spectral`, as in the paper. Without that override, the sampler follows the original general-purpose launcher: AN and APG use hard Stage-1 steering; DAMG, CRD-Reduced, and Cone use soft-spectral steering. `RFM_STAGE1_STEERING_MODE=adaptive_hard` selects hard Stage-1 steering for all methods. APG's Stage-2 projector follows the same choice; DAMG retains its online soft filter.

The main settings are steering strength 10, soft exponent 0.3, 256 preference weights, two noise draws per weight, and a final budget of 256 solutions. Invalid candidates and oracle outputs are filtered, so the returned set can be smaller. RFM geometry is cached; APG also fits Stage-2 geometry. Proxy predictions select candidates, and oracle objectives evaluate them.

CRD-Reduced and Cone use the batched simplex QP solver in `experiments/crd_solver.py` for consensus with more than two objectives and the Cone fallback. The two-objective closed form is retained. For three or more objectives, results can differ from the earlier approximate Frank–Wolfe solver.

## Baselines

### Generative baselines

ParetoFlow+ reuses the models trained by `train_rfm.sh`. ParetoFlow and PCD have their own training pipelines:

```bash
GPUS="0 1 2 3" METHODS="paretoflow_plus" bash scripts/sample_rfm.sh
GPUS="0 1 2 3" ACTION=all bash scripts/run_paretoflow.sh
GPUS="0 1 2 3" ACTION=all bash scripts/run_pcd.sh
```

For ParetoFlow and PCD, set `ACTION=sample` to use existing checkpoints. Add `TASKS="zdt2" SEEDS="0"` to any command for a single-task run.

### DNN surrogate baselines

These commands cover the 11 surrogate baselines in the paper. Each trains a surrogate if its checkpoint is missing, otherwise loads it, then runs NSGA-II:

```bash
GPUS="0 1 2 3" MODEL=end2end \
  TRAIN_MODES="Vallina GradNorm PcGrad" bash scripts/run_surrogate_baselines.sh
GPUS="0 1 2 3" MODEL=multi_head \
  TRAIN_MODES="Vallina GradNorm PcGrad" bash scripts/run_surrogate_baselines.sh
GPUS="0 1 2 3" MODEL=multiple_models \
  TRAIN_MODES="Vallina RoMA IOM ICT TriMentoring" bash scripts/run_surrogate_baselines.sh
```

`Vallina` is the configuration name for the plain model. The default `MODEL=all` runs all 11:

```bash
GPUS="0 1 2 3" bash scripts/run_surrogate_baselines.sh
```

### Other benchmark methods

Multiple Models + COM and GP-based MOBO are additional options, outside the paper's main comparison. PGD-MOO is not included.

```bash
GPUS="0" MODEL=multiple_models TRAIN_MODES="COM" \
  bash scripts/run_surrogate_baselines.sh
python -m pip install -r requirements-mobo.txt
GPUS="0" TASKS="zdt2" SEEDS="0" TRAIN_MODES="Vallina ParEGO JES" \
  bash scripts/run_mobo.sh
```

Start MOBO with a small task, since fitting Gaussian processes can be expensive at the full dataset scale.

## Tasks and launcher options

The benchmark contains **47 tasks**, including Hopper and Swimmer:

| Family | Tasks | Count |
| --- | --- | ---: |
| RE | `re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61` | 15 |
| DTLZ | `dtlz1`–`dtlz7` | 7 |
| ZDT | `zdt1 zdt2 zdt3 zdt4 zdt6` | 5 |
| C-10/MOP | `c10mop1`–`c10mop9` | 9 |
| IN-1K/MOP | `in1kmop1`–`in1kmop9` | 9 |
| MORL | `mo_hopper_v2 mo_swimmer_v2` | 2 |

| Variable | Usage |
| --- | --- |
| `GPUS` | Space-separated GPU indices, e.g. `"0 1"`; one job per selected GPU |
| `TASKS` | Space-separated task names; overrides `TASK_GROUP` |
| `TASK_GROUP` | `all` (default), `non_morl` (45 tasks), or `morl` (2 tasks) |
| `SEEDS` | Space-separated seeds; default `"0 1 2 3 4"` |
| `THREADS` | Numerical-library thread limit per job; default `8` |
| `PYTHON_BIN` | Python interpreter; defaults to the active environment |
| `DRY_RUN` | Set to `1` to print commands without running them |

`non_morl` includes all 18 NAS tasks and therefore still requires the NAS backend. To run only tasks covered by the core installation, give an explicit `TASKS` list, such as `TASKS="re21 dtlz2 zdt2"`.

For example, run one seed of the non-MORL sampling grid in the background:

```bash
mkdir -p logs
nohup env GPUS="0 1 2 3" THREADS=8 TASK_GROUP=non_morl SEEDS="0" \
  RFM_STAGE1_STEERING_MODE=soft_spectral \
  bash scripts/sample_rfm.sh > logs/rfm_sampling.log 2>&1 &
```

## Outputs and tests

RFM models and results are under `experiments/results/`. The default `RUN_TAG=enhanced_training_v1` selects the folders `flow_matching_models_<RUN_TAG>/`, `proxies_model_<RUN_TAG>/`, and `saved_metrics_<RUN_TAG>/`. Keep the same tag for training and sampling.

`RFM_CACHE_DIR` defaults to `experiments/results/rfm_cache_adaptive_variants_v3/` and is independent of `RUN_TAG`. Use a separate cache directory for each trained model set. After retraining or replacing the flow or proxies, choose a new `RFM_CACHE_DIR`, or set both `RETRAIN_RFM=True` and `RETRAIN_RFM_STAGE2=True` when sampling to rebuild the geometry.

Native baseline outputs are under `experiments/offline_moo/`. Per-job logs are under `experiments/logs/public_release/<launcher>/`; set `LOG_DIR` to change the location.

Repeated runs with the same settings overwrite samples and metrics; launcher logs append. Each job log records the command. RFM configuration JSON files are shared across methods for a task and seed, so use the job logs when checking method-specific arguments.

Metrics include HV@100, HV@75, and HV@50, with D(best) as the offline-dataset reference. Keep the same normalization and nadir values when comparing methods. For paper-style summaries, compute ranks per task and seed (average ranks for ties), average within each task family, and report mean and sample standard deviation over the five seeds. Compare sampling times on the same hardware with the same thread limits and output budget; RFM precomputation is timed separately.

Run the numerical tests without downloading datasets:

```bash
python -m unittest discover -s tests -v
```

## Task backends

<details>
<summary><strong>NAS: EvoXBench for C-10/MOP and IN-1K/MOP</strong></summary>

```bash
python -m pip install -r requirements-nas.txt
```

Download the EvoXBench [database](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view) and [model/data assets](https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view). Extract their contents to:

```text
experiments/offline_moo/off_moo_bench/problem/mo_nas/database/
experiments/offline_moo/off_moo_bench/problem/mo_nas/data/
```

Use the inner `data/` folder from `data20240229.zip` (inside `data20240229/`). Under `mo_nas/`, expect `database/ORM/` and `data/nb101/`; avoid an extra nested archive folder. Configure the extracted paths:

```bash
python experiments/offline_moo/config_evoxbench.py
```

These evaluator assets are separate from the offline datasets in `experiments/offline_moo/data/`. See [EvoXBench](https://github.com/EMI-Group/evoxbench) for details.

</details>

<details>
<summary><strong>MORL: MuJoCo for Hopper and Swimmer</strong></summary>

The evaluators require **MuJoCo 2.1.0** with `mujoco-py`. Install the native libraries and Python dependencies in the Conda environment:

```bash
conda install -y -c conda-forge \
  mesalib=25.0.5 libgl-devel glew glfw xorg-libx11 xorg-libxext xorg-libxrender
mkdir -p "$HOME/.mujoco"
curl -fL https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -o /tmp/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install -r requirements-morl.txt
```

Set both environment variables in each shell used for MORL, preserving any NVIDIA driver-library paths supplied by your cluster. Keep GCC 11, Cython 0.29, and the Mesa pin: `mujoco-py` needs OSMesa, which newer Mesa packages no longer provide. `libgl-devel` supplies the linker library. See the [mujoco-py installation notes](https://github.com/openai/mujoco-py#install-mujoco) for host-specific OpenGL setup.

Check physics initialization before launching experiments:

```bash
python - <<'PY'
import mujoco_py
xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
simulation = mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
simulation.step()
print('MuJoCo physics initialization passed')
PY
```

The benchmark data must also include the policy-parameter shape metadata:

```text
experiments/offline_moo/data/mo_hopper_v2/params_shapes.pkl
experiments/offline_moo/data/mo_swimmer_v2/params_shapes.pkl
```

Keep the evaluator assets bundled with the MORL backend. The physics check above does not check these task-specific files.

</details>

## Citation

```bibtex
@article{lu2026learning,
  title   = {Learning Where to Steer: Noise-Space Geometry for Efficient Offline Multi-Objective Optimization with Generative Models},
  author  = {Lu, Yuan and Singh, Esha and Ma, Yi-An and Wang, Yusu},
  journal = {arXiv preprint arXiv:2609.38920},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.38920}
}
```

## Acknowledgements

We build on [Off-MOO-Bench](https://github.com/lamda-bbo/offline-moo), [ParetoFlow](https://github.com/StevenYuan666/ParetoFlow), and [PCD](https://github.com/jatan12/PCD).
