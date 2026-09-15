"""Convert Netgen Neutral meshes to legacy Nmesh HDF5."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..io import save_raw_mesh_as_legacy_nmesh_hdf5
from ..io.netgen import read_netgen_neutral


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nmeshimport",
        description="Convert a Netgen Neutral tetrahedral mesh to Nmesh HDF5.",
    )
    parser.add_argument(
        "--netgen",
        action="store_true",
        help="read Netgen Neutral input (the supported conversion mode)",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.netgen:
        raise SystemExit("nmeshimport currently requires --netgen.")
    mesh = read_netgen_neutral(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_raw_mesh_as_legacy_nmesh_hdf5(args.output, mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
