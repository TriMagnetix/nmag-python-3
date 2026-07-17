#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_BIN="$ROOT_DIR/.venv/bin"
VENV_PYTHON="$VENV_BIN/python"
run_rust=false

usage() {
    printf 'Usage: %s [--rust]\n' "${0##*/}"
}

while (($#)); do
    case "$1" in
        --rust)
            run_rust=true
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

for tool in ruff pyright; do
    if [[ ! -x "$VENV_BIN/$tool" ]]; then
        printf 'Missing %s in .venv. Run ./scripts/setup.sh first.\n' "$tool" >&2
        exit 1
    fi
done

cd "$ROOT_DIR"
printf 'Running Ruff...\n'
if ! ruff_output=$("$VENV_BIN/ruff" check src tests); then
    printf '%s\n' "$ruff_output" >&2
    exit 1
fi

printf 'Running Pyright...\n'
"$VENV_BIN/pyright"

printf 'Running Python tests...\n'
"$VENV_PYTHON" -m pytest

if [[ "$run_rust" == true ]]; then
    if ! command -v cargo >/dev/null 2>&1; then
        printf 'Rust checks requested, but cargo is not installed.\n' >&2
        exit 1
    fi
    printf 'Running Rust formatting check...\n'
    cargo fmt --manifest-path rust/nmag_accel/Cargo.toml --all -- --check

    printf 'Running Rust lint...\n'
    cargo clippy --manifest-path rust/nmag_accel/Cargo.toml --all-targets -- -D warnings

    printf 'Running Rust tests...\n'
    cargo test --manifest-path rust/nmag_accel/Cargo.toml
fi

printf 'All requested checks passed.\n'
