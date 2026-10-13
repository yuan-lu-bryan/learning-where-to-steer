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
python -m pip install numpy==1.23.2 scipy==1.10.1 Cython==0.29.37
python -m pip install --no-use-pep517 --no-build-isolation scikit-learn==0.21.3
python -m pip install -r requirements.txt
python -m pip check
```

Complete the [scikit-learn compatibility step](#scikit-learn-compatibility) below after installation. MuJoCo is only needed for Hopper and Swimmer.

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

To run the four RFM methods on all 47 tasks with seeds `0 1 2 3 4`, first complete the NAS and MORL setup below:

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

The paper uses `soft_spectral` steering. By default, AN and APG use hard Stage-1 steering; DAMG, CRD-Reduced, and Cone use soft-spectral steering. Set `RFM_STAGE1_STEERING_MODE=adaptive_hard` to use hard Stage-1 steering for all methods. APG's Stage-2 projector uses the same mode; DAMG uses an online soft filter.

The main settings are steering strength 10, soft exponent 0.3, 256 preference weights, two noise draws per weight, and a final budget of 256 solutions. Invalid candidates and oracle outputs are filtered, so the returned set can be smaller. RFM geometry is cached; APG also fits Stage-2 geometry. Proxy predictions select candidates, and oracle objectives evaluate them.

CRD-Reduced and Cone use a batched simplex QP solver for consensus with more than two objectives and for the Cone fallback. Two-objective consensus uses a closed-form solution.

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

### MORL

<details>
<summary>Hopper and Swimmer setup</summary>

#### scikit-learn compatibility

The shared environment needs this fix for `scikit-learn==0.21.3` with SciPy 1.10.1:

```bash
python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

package = distribution("scikit-learn")
assert package.version == "0.21.3", package.version
path = Path(package.locate_file("sklearn/cross_decomposition/pls_.py"))
original = path.read_text()
patched = original.replace("pinv2", "pinv")
assert "from scipy.linalg import pinv, svd" in patched, "Unexpected sklearn PLS source"
if patched != original:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(patched)
print("sklearn PLS ready:", path)
PY
python - <<'PY'
import scipy.linalg
import sklearn
from sklearn.cross_decomposition import pls_
from sklearn.model_selection import train_test_split

assert sklearn.__version__ == "0.21.3", sklearn.__version__
assert pls_.pinv is scipy.linalg.pinv
print("sklearn import and PLS compatibility check passed")
PY
python -m pip check
```

#### MuJoCo

The evaluators require **MuJoCo 2.1.0** with `mujoco-py`. Install the native libraries and Python dependencies in the Conda environment:

```bash
conda install -y -c conda-forge \
  mesalib=25.0.5 libgl-devel glew glfw xorg-libx11 xorg-libxext xorg-libxrender
mkdir -p "$HOME/.mujoco"
curl -fL https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -o /tmp/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"
python -m pip install -r requirements-morl.txt
```

Set the library paths in each shell used for MORL:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
MUJOCO_NVIDIA_LIB_DIR=$(python - <<'PY'
import glob
import os
import shutil

path = ""
if shutil.which("nvidia-smi"):
    for candidate in ("/usr/local/nvidia/lib64", "/usr/lib/nvidia"):
        if os.path.isdir(candidate):
            path = candidate
            break
    if not path:
        candidates = sorted(glob.glob("/usr/lib/nvidia-[0-9][0-9][0-9]"))
        if candidates:
            path = candidates[-1]
print(path)
PY
)
export LD_LIBRARY_PATH="${MUJOCO_NVIDIA_LIB_DIR:+$MUJOCO_NVIDIA_LIB_DIR:}$MUJOCO_PY_MUJOCO_PATH/bin:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The pinned Mesa package provides OSMesa for CPU rendering. See [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco) for host-specific OpenGL setup.

For NVIDIA GPUs, apply this EGL fix before importing `mujoco_py`:

```bash
python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

package = distribution("mujoco-py")
assert package.version == "2.1.2.14", package.version
path = Path(package.locate_file("mujoco_py/gl/eglshim.c"))
original = path.read_text()
text = original
anchor = '#include "glshim.h"\n'
assert text.count(anchor) == 1, "Unexpected EGL include layout"
headers = ("stdio.h", "string.h", "stdint.h")
missing = [h for h in headers if f"#include <{h}>" not in text]
text = text.replace(anchor, anchor + "".join(f"#include <{h}>\n" for h in missing))
replacements = [
    ('fprintf("glewInit error: %s\\n", glewGetErrorString(err));',
     'fprintf(stderr, "glewInit error: %s\\n", glewGetErrorString(err));'),
    ('bufferOffset * viewport.width * viewport.height * 3);',
     '(void *)(uintptr_t)(bufferOffset * viewport.width * viewport.height * 3));'),
    ('bufferOffset * viewport.width * viewport.height * sizeof(short));',
     '(void *)(uintptr_t)(bufferOffset * viewport.width * viewport.height * sizeof(short)));'),
]
for old, new in replacements:
    if text.count(old) == 1 and new not in text:
        text = text.replace(old, new)
    else:
        assert text.count(new) == 1, "Unexpected EGL source: " + old
if text != original:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(text)
print("EGL shim ready:", path)
PY
```

Build the MuJoCo extension:

```bash
CFLAGS="${CFLAGS-} -Wno-error=incompatible-pointer-types" \
CXXFLAGS="${CXXFLAGS-} -Wno-error=incompatible-pointer-types" \
MUJOCO_PY_FORCE_REBUILD=1 \
  python -c "import mujoco_py; print('mujoco_py import OK:', mujoco_py.cymj.__file__)"
```

Check physics initialization:

```bash
python - <<'PY'
import mujoco_py
xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
simulation = mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
simulation.step()
print('MuJoCo physics initialization passed')
PY
```

Check offscreen rendering on an NVIDIA GPU with `MUJOCO_PY_FORCE_CPU` unset. Set `device_id` to your EGL device index, which may differ from PyTorch's GPU numbering:

```bash
python - <<'PY'
import mujoco_py

assert "linuxgpuextensionbuilder" in mujoco_py.cymj.__file__, "CPU extension loaded"
xml = '<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
simulation = mujoco_py.MjSim(mujoco_py.load_model_from_xml(xml))
simulation.step()
context = mujoco_py.MjRenderContextOffscreen(simulation, device_id=0)
pixels = simulation.render(64, 64)
assert pixels.shape == (64, 64, 3), pixels.shape
print("GPU EGL offscreen rendering passed")
PY
```

The MORL data folders must include:

```text
experiments/offline_moo/data/mo_hopper_v2/params_shapes.pkl
experiments/offline_moo/data/mo_swimmer_v2/params_shapes.pkl
```

#### Optional tools

For notebooks and additional utilities:

```bash
python -m pip install -c requirements.txt ipykernel==6.29.5 swanlab==0.7.20
python -m pip install -c requirements.txt --no-build-isolation POT==0.9.5
python -m pip install --no-deps opencv-python-headless==4.8.1.78
python -m pip check
```

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
