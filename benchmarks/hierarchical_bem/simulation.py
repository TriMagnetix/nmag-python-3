from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

import nmag
import nmesh
from nmag.config import DemagBemStorage

from .meshes import structured_thin_film_mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run static and fixed-time demag on a generated thin film."
    )
    parser.add_argument("--boundary-nodes", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--target-time", type=float, default=1.0e-15)
    parser.add_argument("--exact-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _create_mesh(path: Path, target_boundary_nodes: int) -> tuple[int, int]:
    generated = structured_thin_film_mesh(target_boundary_nodes)
    mesh = nmesh.mesh_from_points_and_simplices(
        points=generated.points.tolist(),
        simplices_indices=generated.simplices.tolist(),
        simplices_regions=[1] * len(generated.simplices),
    )
    mesh.save(path)
    return len(generated.points), len(generated.simplices)


def _simulation(
    mesh_path: Path,
    output_directory: Path,
    *,
    storage: DemagBemStorage,
    memory_fraction: float,
) -> nmag.Simulation:
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    simulation = nmag.Simulation(
        name=f"thin-film-{storage}",
        config=nmag.NmagConfig(
            output_directory=output_directory,
            output_policy="replace",
            accelerator="rust",
            demag_bem_storage=storage,
            hierarchical_bem=nmag.HierarchicalBemConfig(memory_fraction=memory_fraction),
        ),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("film", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, 0.02 * point[0], -0.02 * point[1]])
    return simulation


def _static_values(simulation: nmag.Simulation) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(simulation.get_subfield("H_demag"), dtype=np.float64)
    energy = np.asarray(simulation.get_subfield("E_demag"), dtype=np.float64)
    return field, energy


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        cast(Any, np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-30))
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if args.target_time <= 0.0:
        raise ValueError("target_time must be positive")

    with tempfile.TemporaryDirectory(prefix="nmag-hmatrix-film-") as directory:
        work = Path(directory)
        mesh_path = work / "thin-film.nmesh.h5"
        points, tetrahedra = _create_mesh(mesh_path, args.boundary_nodes)

        simulation = _simulation(
            mesh_path,
            work / "hierarchical",
            storage="hierarchical",
            memory_fraction=args.memory_fraction,
        )
        static_started = time.perf_counter()
        field, energy = _static_values(simulation)
        static_seconds = time.perf_counter() - static_started
        stats = simulation.last_bem_operator_stats
        if stats is None:
            raise RuntimeError("hierarchical BEM did not report operator diagnostics")

        reference_seconds: float | None = None
        field_error: float | None = None
        energy_error: float | None = None
        if args.exact_reference:
            reference = _simulation(
                mesh_path,
                work / "matrix-free",
                storage="matrix-free",
                memory_fraction=args.memory_fraction,
            )
            reference_started = time.perf_counter()
            reference_field, reference_energy = _static_values(reference)
            reference_seconds = time.perf_counter() - reference_started
            field_error = _relative_error(field, reference_field)
            energy_error = _relative_error(energy, reference_energy)

        dynamics_started = time.perf_counter()
        simulation.advance_time(nmag.SI(args.target_time, "s"))
        dynamics_seconds = time.perf_counter() - dynamics_started
        magnetization = np.asarray(simulation.get_subfield("m"), dtype=np.float64)

        return {
            "requested_boundary_nodes": args.boundary_nodes,
            "points": points,
            "tetrahedra": tetrahedra,
            "effective_backend": stats.effective_backend,
            "fallback_reason": stats.fallback_reason,
            "setup_and_static_seconds": static_seconds,
            "exact_reference_seconds": reference_seconds,
            "field_relative_error": field_error,
            "energy_relative_error": energy_error,
            "fixed_time_target_seconds": args.target_time,
            "fixed_time_actual_seconds": simulation.clock.time_reached_si.in_units_of(
                nmag.SI(1.0, "s")
            ),
            "fixed_time_dynamics_seconds": dynamics_seconds,
            "accepted_steps": simulation.last_integrator_stats.accepted_steps,
            "finite_field": bool(np.isfinite(field).all()),
            "finite_energy": bool(np.isfinite(energy).all()),
            "finite_magnetization": bool(np.isfinite(magnetization).all()),
            "maximum_magnetization_norm_error": float(
                np.max(np.abs(np.linalg.norm(magnetization, axis=1) - 1.0))
            ),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "bem": {
                "storage_bytes": stats.storage_bytes,
                "dense_equivalent_bytes": stats.dense_equivalent_bytes,
                "compression_ratio": stats.compression_ratio,
                "dense_blocks": stats.dense_blocks,
                "low_rank_blocks": stats.low_rank_blocks,
                "maximum_rank": stats.maximum_rank,
                "mean_rank": stats.mean_rank,
                "sampled_relative_error": stats.sampled_relative_error,
            },
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
