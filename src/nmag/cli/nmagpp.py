"""Export modern Nmag snapshots to VTK-compatible files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import h5py

from ..vtk_export import export_vtk, resolve_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nmagpp",
        description="Export a modern Nmag HDF5 snapshot to VTK.",
    )
    parser.add_argument("--vtk", dest="output", required=True, type=Path)
    parser.add_argument("input", type=Path, help="snapshot path or simulation base name")
    parser.add_argument("--mesh", type=Path, help="mesh file supplying cell topology")
    parser.add_argument("--field", action="append", dest="fields", metavar="NAME")
    parser.add_argument("--all-fields", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = resolve_snapshot(args.input)
    if args.all_fields:
        with h5py.File(str(snapshot), "r") as handle:
            group = handle.get("fields")
            if not isinstance(group, h5py.Group):
                raise ValueError("Snapshot is missing the /fields group.")
            fields = tuple(str(name) for name in group.keys())
    else:
        fields = tuple(args.fields or ("m",))
    export_vtk(snapshot, args.output, mesh_path=args.mesh, fields=fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
