from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import nmag
from nmag.simulation.demag.bem.hierarchical import build_hierarchical_lindholm_operator

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE_MESH = FIXTURES / "nmag_doc_example1" / "sphere1.nmesh.h5"


def _simulation(
    tmp_path: Path,
    *,
    storage: Literal["auto", "dense", "hierarchical", "matrix-free"],
    accelerator: Literal["auto", "off", "rust"] = "rust",
    hierarchical_bem: nmag.HierarchicalBemConfig | None = None,
) -> nmag.Simulation:
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    simulation = nmag.Simulation(
        name=f"bem-{storage}",
        config=nmag.NmagConfig(
            output_directory=tmp_path,
            demag_bem_storage=storage,
            accelerator=accelerator,
            hierarchical_bem=hierarchical_bem or nmag.HierarchicalBemConfig(),
        ),
    )
    simulation.load_mesh(
        str(SPHERE_MESH),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m(lambda point: [1.0, 0.1 * point[0], -0.05 * point[1]])
    return simulation


def test_hierarchical_demag_matches_dense_and_reports_diagnostics(tmp_path: Path) -> None:
    pytest.importorskip("nmag_accel")
    dense = _simulation(tmp_path, storage="dense")
    hierarchical = _simulation(tmp_path, storage="hierarchical")

    np.testing.assert_allclose(
        hierarchical.get_subfield("H_demag"),
        dense.get_subfield("H_demag"),
        rtol=1.0e-8,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        hierarchical.get_subfield("E_demag"),
        dense.get_subfield("E_demag"),
        rtol=1.0e-8,
        atol=1.0e-5,
    )
    stats = hierarchical.last_bem_operator_stats
    assert stats is not None
    assert stats.requested_backend == "hierarchical"
    assert stats.effective_backend == "hierarchical"
    assert stats.fallback_reason is None
    assert stats.sampled_relative_error <= 1.0e-6
    assert stats.storage_bytes > 0
    with pytest.raises(AttributeError):
        hierarchical.last_bem_operator_stats = None  # type: ignore[misc]
    hierarchical._invalidate_demag(clear_geometry=True)
    assert hierarchical.last_bem_operator_stats is None


def test_hierarchical_mode_falls_back_exactly_when_rust_is_disabled(tmp_path: Path) -> None:
    simulation = _simulation(tmp_path, storage="hierarchical", accelerator="off")
    field = np.asarray(simulation.get_subfield("H_demag"))

    assert np.isfinite(field).all()
    stats = simulation.last_bem_operator_stats
    assert stats is not None
    assert stats.effective_backend == "matrix-free"
    assert "unavailable or disabled" in (stats.fallback_reason or "")


def test_hierarchical_mode_falls_back_when_memory_budget_is_exhausted(tmp_path: Path) -> None:
    pytest.importorskip("nmag_accel")
    simulation = _simulation(
        tmp_path,
        storage="hierarchical",
        hierarchical_bem=nmag.HierarchicalBemConfig(memory_fraction=1.0e-15),
    )
    field = np.asarray(simulation.get_subfield("H_demag"))

    assert np.isfinite(field).all()
    stats = simulation.last_bem_operator_stats
    assert stats is not None
    assert stats.effective_backend == "matrix-free"
    assert "memory budget exceeded" in (stats.fallback_reason or "")


def test_hierarchical_operator_rejects_a_wrong_vector_shape(tmp_path: Path) -> None:
    pytest.importorskip("nmag_accel")
    simulation = _simulation(tmp_path, storage="hierarchical")
    points = np.asarray(simulation.mesh.points, dtype=float)
    simplices = np.asarray(simulation.mesh.simplices, dtype=np.int64)
    faces = simulation._boundary_faces_for_demag_mesh(points, simplices)
    _nodes, operator, fallback = build_hierarchical_lindholm_operator(
        simulation,
        points,
        simplices,
        faces,
    )

    assert fallback is None
    with pytest.raises(ValueError, match="BEM input must have shape"):
        operator @ np.zeros(operator.shape[1] - 1)
    values = np.zeros(operator.shape[1])
    values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        operator @ values


def test_hierarchical_fixed_time_dynamics_match_dense(tmp_path: Path) -> None:
    pytest.importorskip("nmag_accel")
    dense = _simulation(tmp_path, storage="dense")
    hierarchical = _simulation(tmp_path, storage="hierarchical")
    target = nmag.SI(2.0e-14, "s")

    dense.advance_time(target)
    hierarchical.advance_time(target)

    assert hierarchical.clock.time_reached_si == dense.clock.time_reached_si
    np.testing.assert_allclose(
        hierarchical.get_subfield("m"),
        dense.get_subfield("m"),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
