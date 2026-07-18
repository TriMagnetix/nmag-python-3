from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any, cast

import nmag_accel
import numpy as np

from nmag.resources import available_memory_bytes

from .meshes import OperatorMesh, fibonacci_sphere_mesh, structured_thin_film_mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark certified hierarchical Lindholm BEM.")
    parser.add_argument("--geometry", choices=("sphere", "thin-film"), default="sphere")
    parser.add_argument("--boundary-nodes", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--matvec-repeats", type=int, default=3)
    parser.add_argument("--exact-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _build_geometry(name: str, boundary_nodes: int) -> OperatorMesh:
    if name == "sphere":
        return fibonacci_sphere_mesh(boundary_nodes)
    return structured_thin_film_mesh(boundary_nodes)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    mesh_started = time.perf_counter()
    mesh = _build_geometry(args.geometry, args.boundary_nodes)
    accelerator = cast(Any, nmag_accel)
    _owners, raw_faces = accelerator.build_oriented_boundary_faces(
        mesh.points,
        mesh.simplices,
    )
    faces = np.ascontiguousarray(raw_faces, dtype=np.int64)
    boundary_nodes = np.unique(faces)
    local_index_by_point = np.full(len(mesh.points), -1, dtype=np.int64)
    local_index_by_point[boundary_nodes] = np.arange(len(boundary_nodes), dtype=np.int64)
    mesh_seconds = time.perf_counter() - mesh_started

    available = available_memory_bytes()
    memory_budget = (
        int(available * args.memory_fraction) if available is not None else np.iinfo(np.uintp).max
    )
    setup_started = time.perf_counter()
    operator = accelerator.build_lindholm_hmatrix(
        mesh.points,
        mesh.simplices,
        faces,
        boundary_nodes,
        local_index_by_point,
        memory_budget_bytes=memory_budget,
    )
    setup_seconds = time.perf_counter() - setup_started

    vector = np.random.default_rng(20260717).standard_normal(len(boundary_nodes))
    timings: list[float] = []
    result = np.empty_like(vector)
    for _ in range(args.matvec_repeats):
        started = time.perf_counter()
        result = np.asarray(operator.matvec(vector), dtype=np.float64)
        timings.append(time.perf_counter() - started)

    exact_relative_error: float | None = None
    exact_seconds: float | None = None
    if args.exact_reference:
        exact_started = time.perf_counter()
        exact = np.asarray(
            accelerator.apply_lindholm_bem_matrix_free(
                mesh.points,
                mesh.simplices,
                faces,
                boundary_nodes,
                local_index_by_point,
                vector,
            ),
            dtype=np.float64,
        )
        exact_seconds = time.perf_counter() - exact_started
        exact_relative_error = float(
            cast(Any, np.linalg.norm(result - exact) / np.linalg.norm(exact))
        )

    dense_bytes = len(boundary_nodes) ** 2 * np.dtype(np.float64).itemsize
    return {
        "geometry": args.geometry,
        "requested_boundary_nodes": args.boundary_nodes,
        "boundary_nodes": len(boundary_nodes),
        "boundary_faces": len(faces),
        "tetrahedra": len(mesh.simplices),
        "mesh_seconds": mesh_seconds,
        "setup_seconds": setup_seconds,
        "matvec_seconds": timings,
        "exact_reference_seconds": exact_seconds,
        "exact_relative_error": exact_relative_error,
        "finite_output": bool(np.isfinite(result).all()),
        "memory_budget_bytes": memory_budget,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "storage_bytes": int(operator.storage_bytes),
        "dense_equivalent_bytes": dense_bytes,
        "compression_ratio": float(operator.storage_bytes / dense_bytes),
        "dense_blocks": int(operator.dense_blocks),
        "low_rank_blocks": int(operator.low_rank_blocks),
        "maximum_rank": int(operator.maximum_rank),
        "mean_rank": float(operator.mean_rank),
        "sampled_relative_error": float(operator.sampled_relative_error),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rust_api_version": int(accelerator.API_VERSION),
    }


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
