# DAFx Challenge 2026 – Task A

Differentiable modal plate model for impulse response estimation.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Select the target IR

Open [Model/ParamsEstimation.py](Model/ParamsEstimation.py) and set `target_npz_path` to the `.npz` file you want to fit:

```python
target_npz_path = "target/2026-DATASET-STRIPPED/random_IR_00XX.npz"
```

The available IRs are in `target/2026-DATASET-STRIPPED/`.

### 2. Run the estimation

```bash
python Model/ParamsEstimation.py
```

The estimated parameters are saved to `experiment_results_taskA/best_params_<index>.csv`.
