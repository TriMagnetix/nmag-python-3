#!/usr/bin/env python3
"""Run modern nmesh scenarios and compare them with legacy mesh artifacts.

Examples:
    python tools/nmesh_parity_compare.py --write-new --output-dir parity/new
    python tools/nmesh_parity_compare.py --legacy-dir parity/legacy
    python tools/nmesh_parity_compare.py --legacy-command "legacy_runner {scenario} {output}"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import nmesh
from nmesh.backend import RawMesh
from nmesh.mesher.parity import (
    assert_canonical_mesh_equal,
    compare_mesh_metrics,
    mesh_metric_summary,
    read_ascii_nmesh,
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One modern-vs-legacy nmesh comparison case."""

    name: str
    build: Callable[[], RawMesh]


def _box2d() -> RawMesh:
    return nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.1, 0.1], [0.9, 0.9])],
        a0=0.35,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    ).raw_mesh


def _box3d() -> RawMesh:
    return nmesh.Mesh(
        bounding_box=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        objects=[nmesh.Box([0.2, 0.2, 0.2], [0.8, 0.8, 0.8])],
        a0=0.3,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    ).raw_mesh


def _adjacent2d() -> RawMesh:
    return nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[
            nmesh.Box([0.0, 0.0], [0.5, 1.0]),
            nmesh.Box([0.5, 0.0], [1.0, 1.0]),
        ],
        a0=0.5,
        max_steps=20,
        nr_probes_for_determining_volume=500,
    ).raw_mesh


def _concave2d() -> RawMesh:
    return nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[
            nmesh.difference(
                nmesh.Box([0.0, 0.0], [1.0, 1.0]),
                [nmesh.Box([0.35, 0.35], [0.65, 0.65])],
            )
        ],
        a0=0.25,
        max_steps=20,
        nr_probes_for_determining_volume=1000,
    ).raw_mesh


def _periodic2d() -> RawMesh:
    return nmesh.Mesh(
        bounding_box=[[0.0, 0.0], [1.0, 1.0]],
        objects=[nmesh.Box([0.0, 0.0], [1.0, 1.0])],
        mesh_bounding_box=True,
        periodic=[True, True],
        a0=0.4,
        max_steps=20,
        nr_probes_for_determining_volume=600,
    ).raw_mesh


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario("box2d", _box2d),
        Scenario("box3d", _box3d),
        Scenario("adjacent2d", _adjacent2d),
        Scenario("concave2d", _concave2d),
        Scenario("periodic2d", _periodic2d),
    )
}


def main() -> int:
    """Run configured scenarios and print a JSON parity report."""

    args = _parse_args()
    scenario_names = args.scenario or sorted(SCENARIOS)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nmesh-parity-") as temp_dir:
        temp_path = Path(temp_dir)
        report = []
        exit_code = 0
        for scenario_name in scenario_names:
            scenario = SCENARIOS[scenario_name]
            modern_mesh = scenario.build()
            modern_path = (output_dir or temp_path) / f"{scenario_name}.modern.nmesh"
            nmesh.write_mesh(modern_mesh, modern_path)

            scenario_report = {
                "scenario": scenario_name,
                "modern_path": str(modern_path),
                "modern_metrics": asdict(mesh_metric_summary(modern_mesh)),
            }
            legacy_path = _legacy_path_for(args, scenario_name, temp_path)
            if legacy_path is None:
                scenario_report["status"] = "modern-written"
                report.append(scenario_report)
                continue

            legacy_mesh = read_ascii_nmesh(legacy_path)
            scenario_report["legacy_path"] = str(legacy_path)
            scenario_report["legacy_metrics"] = asdict(mesh_metric_summary(legacy_mesh))
            exact_failure = _exact_failure(modern_mesh, legacy_mesh, args.coordinate_tolerance)
            comparison = compare_mesh_metrics(
                modern_mesh,
                legacy_mesh,
                count_relative_tolerance=args.count_relative_tolerance,
                volume_relative_tolerance=args.volume_relative_tolerance,
                length_relative_tolerance=args.length_relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
            )
            scenario_report["exact_match"] = exact_failure is None
            if exact_failure is not None:
                scenario_report["exact_failure"] = exact_failure
            scenario_report["metric_passed"] = comparison.passed
            scenario_report["metric_failures"] = list(comparison.failures)
            scenario_report["status"] = "passed" if comparison.passed else "failed"
            if not comparison.passed:
                exit_code = 1
            report.append(scenario_report)

    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Scenario to run. May be provided multiple times. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        help="Directory containing legacy '<scenario>.nmesh' or '<scenario>.legacy.nmesh' files.",
    )
    parser.add_argument(
        "--legacy-command",
        help=(
            "Command template used to generate a legacy mesh. The template may "
            "contain {scenario} and {output}; it is split with shlex and run without a shell."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where modern scenario meshes should be written.",
    )
    parser.add_argument("--coordinate-tolerance", type=float)
    parser.add_argument("--count-relative-tolerance", type=float, default=0.25)
    parser.add_argument("--volume-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--length-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-9)
    return parser.parse_args()


def _legacy_path_for(
    args: argparse.Namespace,
    scenario_name: str,
    temp_path: Path,
) -> Path | None:
    if args.legacy_command:
        output_path = temp_path / f"{scenario_name}.legacy.nmesh"
        command = args.legacy_command.format(
            scenario=scenario_name,
            output=str(output_path),
        )
        subprocess.run(shlex.split(command), check=True)
        return output_path

    if args.legacy_dir is None:
        return None

    candidates = [
        args.legacy_dir / f"{scenario_name}.legacy.nmesh",
        args.legacy_dir / f"{scenario_name}.nmesh",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No legacy mesh found for {scenario_name!r}; expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _exact_failure(
    actual: RawMesh,
    expected: RawMesh,
    coordinate_tolerance: float | None,
) -> str | None:
    try:
        assert_canonical_mesh_equal(
            actual,
            expected,
            coordinate_tolerance=coordinate_tolerance,
        )
    except AssertionError as exc:
        return str(exc)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
