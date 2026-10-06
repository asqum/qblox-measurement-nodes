# Installing on Linux ARM64 (no sudo)

Use micromamba to create a private Python environment when you do not have sudo.
The package supports Python 3.10–3.12.

## 1. Install micromamba

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
micromamba --version
```

## 2. Create and activate the environment

```bash
micromamba create -n qblox python=3.10 pip -c conda-forge
micromamba activate qblox
which python
python --version
```

## 3. Install this repository

```bash
git clone https://github.com/asqum/qblox-measurement-nodes.git
cd qblox-measurement-nodes
python -m pip install -e .
```

## 4. Verify

```bash
python -c "import qblox_scheduler; print(qblox_scheduler.__version__)"
python -c "import qblox_instruments; print(qblox_instruments.__version__)"
python -m pip check
```

If `pip check` reports no errors, the environment is ready. A working installation
had `qblox-instruments 1.3.0`, `qblox-scheduler 1.0.0b6` and `quantify-core 0.8.3`.

## Using the environment

```bash
source ~/.bashrc            # once per shell, to initialise micromamba
micromamba activate qblox
micromamba deactivate       # to leave
```
