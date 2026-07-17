#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
MANIFEST_PATH="$ROOT_DIR/rust/nmag_accel/Cargo.toml"
build_mode=debug

usage() {
    printf 'Usage: %s [--release]\n' "${0##*/}"
}

while (($#)); do
    case "$1" in
        --release)
            build_mode=release
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! -x "$VENV_PYTHON" ]]; then
    printf 'Missing %s. Run ./scripts/setup.sh first.\n' "$VENV_PYTHON" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -c 'import maturin' >/dev/null 2>&1; then
    printf 'maturin is not installed in .venv. Run ./scripts/setup.sh or install .[rust].\n' >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
    printf 'Rust is required for the accelerator. Install it with rustup, then rerun this script.\n' >&2
    exit 1
fi

cd "$ROOT_DIR"
arguments=(develop --manifest-path "$MANIFEST_PATH")
if [[ "$build_mode" == release ]]; then
    arguments+=(--release)
fi

"$VENV_PYTHON" -m maturin "${arguments[@]}"
