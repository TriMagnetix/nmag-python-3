#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN=${PYTHON:-python3}

cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    printf 'Creating virtual environment in %s\n' "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip

require_python_development_library() {
    local library_path dev_package
    library_path=$("$VENV_PYTHON" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR") or "", sysconfig.get_config_var("LDLIBRARY") or ""))')

    # PyO3 links the accelerator against libpython.  Runtime Python packages
    # commonly provide only libpythonX.Y.so.1.0; the unversioned linker name is
    # supplied by the matching development package.
    if [[ -z "$library_path" || ! -e "$library_path" ]]; then
        dev_package=$("$VENV_PYTHON" -c 'import sys; print(f"libpython{sys.version_info.major}.{sys.version_info.minor}-dev")')
        printf 'Python development library not found: %s\n' "$library_path" >&2
        printf 'The Rust accelerator needs the matching development package. On Debian/Ubuntu run:\n' >&2
        printf '  sudo apt install %s\n' "$dev_package" >&2
        exit 1
    fi
}

build_accelerator=${NMAG_SETUP_ACCELERATOR:-}
if [[ -z "$build_accelerator" && -t 0 ]]; then
    read -r -p 'Build the optional Rust accelerator? [y/N] ' build_accelerator
fi

case "${build_accelerator,,}" in
    y|yes)
        if ! command -v cargo >/dev/null 2>&1; then
            printf 'Rust is required for the accelerator. Install it with rustup, then rerun setup.\n' >&2
            exit 1
        fi
        require_python_development_library
        # rustup installs `cargo` as a proxy.  On machines without a configured
        # default toolchain, selecting one explicitly keeps maturin and the
        # subsequent build/check commands from failing before Rust starts.
        export RUSTUP_TOOLCHAIN="${NMAG_RUST_TOOLCHAIN:-stable}"
        printf 'Using Rust toolchain: %s\n' "$RUSTUP_TOOLCHAIN"
        "$VENV_PYTHON" -m pip install -e '.[dev,rust]'
        "$ROOT_DIR/scripts/build-accelerator.sh" --release
        "$ROOT_DIR/scripts/verify.sh" --rust
        ;;
    *)
        "$VENV_PYTHON" -m pip install -e '.[dev]'
        "$ROOT_DIR/scripts/verify.sh"
        ;;
esac
