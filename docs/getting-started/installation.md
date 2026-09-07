# Installation

## Requirements

Nmag currently targets Linux and Python 3.10 or newer. A normal installation
needs:

- Python, `venv`, and `pip`;
- a C compiler for dependencies without a wheel for your platform; and
- Git to work from the source repository.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential
```

## Recommended setup

Clone the repository and use its setup script:

```bash
git clone https://github.com/TriMagnetix/nmag-python-3.git
cd nmag-python-3
./scripts/setup.sh
```

The script creates or reuses `.venv`, installs Nmag in editable mode, offers to
build the optional Rust accelerator, and runs the standard checks. It invokes
the virtual environment explicitly, so activation is not required.

Run a simulation with:

```bash
.venv/bin/python simulation.py
```

You may activate the environment if that is more convenient:

```bash
source .venv/bin/activate
python simulation.py
```

## Manual setup

For CI or an existing environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Normal simulations do not require Rust. To build the optional accelerator,
install Rust with [rustup](https://rustup.rs/), install the `rust` extra, and run
the build script:

```bash
.venv/bin/python -m pip install -e '.[rust]'
./scripts/build-accelerator.sh --release
```

Python 3.12 accelerator builds on Ubuntu or Debian also need
`libpython3.12-dev`. Substitute the development package matching your Python
version.

## Verify the installation

```bash
.venv/bin/python -c "import nmag; print(nmag.SI(1, 'A/m'))"
```

Then run the [quickstart](quickstart.md). If the optional accelerator was built,
`nmag.parallel_runtime_info()` reports its effective thread configuration.

## Building this documentation

Documentation tools are isolated in a separate extra:

```bash
.venv/bin/python -m pip install -e '.[docs]'
.venv/bin/mkdocs serve
```

Use `mkdocs build --strict` before submitting documentation changes.
