# Installation

Use Python 3.11. The demo runs on a CPU with the model, inputs and results included in the repository. It has been run on Windows and in the Python 3.11 Docker environment on GitHub Actions (Ubuntu 24.04).

## Set up an environment

```shell
python -m venv .venv
```

In Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`. On Linux or macOS, use `source .venv/bin/activate`.

```shell
python -m pip install -r requirements.txt
python run.py
```

Package installation requires an internet connection. The demo can then run offline. Allow about 1 GB of disk space and start with 4 GB of RAM for the demo. Its measured runtime and installed package versions are written to `results/execution.json`. Full training can take hours or days.

## Research dependencies

```shell
python -m pip install -r src/requirements-full.txt
python run.py --check-inputs
```

See the [reproduction guide](REPRODUCE.md) for training and analysis commands. Record the environment for a new training run with `python -m pip freeze`.

| Message | Resolution |
|---|---|
| Choose a fresh or empty output directory | Use `--output-dir another_results` |
| Bundled model or archive missing | Download the complete repository, including `data/` |
| Checksum mismatch | Restore the file from the same version |
| Missing research dependency | Install `src/requirements-full.txt` in the active environment |
| Windows activation blocked | Use `.venv\Scripts\python.exe` directly |
