# Installation

Python 3.11 and a CPU are sufficient for the demonstration, saved-result checks and plotting. No GPU, external service or paid API is used. GitHub Actions runs on Ubuntu 24.04 in the supplied Python 3.11 Docker image. Local checks were also run on Windows with Python 3.11.9. macOS execution has not been tested.

Allow approximately 1 GB free disk space for the repository, environment and outputs. For the demo, 4 GB RAM is a practical starting allocation rather than a measured minimum. Full training requires substantially more memory and time, depending on thread settings and retained models. No unmeasured minimum is claimed.

## Install

Create a fresh environment:

```shell
python -m venv .venv
```

Activate it on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on Linux/macOS with `source .venv/bin/activate`. Then:

```shell
python -m pip install -r requirements.txt
python run.py
```

The requirements file pins six direct review dependencies. Exact installed versions and Python/platform details are recorded with each execution. The Docker base tracks Python 3.11 and is not presented as an immutable image digest.

Installation commonly takes a few minutes when binary wheels are available; this is an estimate affected by network speed and caches. The earlier prediction-and-check GitHub workflow took 45 seconds including overhead. The expanded local plotting/demo run took approximately 135 seconds on Windows/Python 3.11.9 and records its measured duration in `results/execution.json`. Full training can take hours or days.

## Research dependencies

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
```

Research workflows additionally use LightGBM, SHAP, imbalanced-learn and statsmodels. Recorded benchmark versions are retained. Historical versions of all scorer/prenatal dependencies were not recorded and are not guessed. Retain `python -m pip freeze` alongside any new training outputs.

## Offline operation

Installation needs package-repository access. Data, ontologies, model and frozen results are bundled. Execution needs no network. GitHub runs the checks and plotting in Docker with networking disabled.

| Message | Resolution |
|---|---|
| Choose a fresh or empty output directory | Use `--output-dir another_results` |
| Bundled model/archive missing | Download the complete repository including `data/` |
| Checksum mismatch | Restore that file from the same release |
| Missing research dependency | Install `src/requirements-full.txt` in the active environment |
| Windows activation blocked | Use `.venv\Scripts\python.exe` directly |
