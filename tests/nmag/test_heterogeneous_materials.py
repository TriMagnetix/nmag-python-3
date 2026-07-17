from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmesh
from nmag.simulation.support import MU0


def _write_disconnected_two_region_mesh(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [10.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        simplices_regions=[1, 2],
    )
    mesh.save(str(path))


def _write_shared_two_region_mesh(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3], [1, 2, 3, 4]],
        simplices_regions=[1, 2],
    )
    mesh.save(str(path))


def _material(
    name: str,
    *,
    ms: float = 1.0e6,
    exchange: float = 0.0,
    damping: float = 0.5,
    precession: bool = False,
) -> nmag.MagMaterial:
    return nmag.MagMaterial(
        name=name,
        Ms=nmag.SI(ms, "A/m"),
        exchange_coupling=nmag.SI(exchange, "J/m"),
        llg_damping=damping,
        do_precession=precession,
    )


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    first: nmag.MagMaterial,
    second: nmag.MagMaterial,
    shared: bool = False,
    config: nmag.NmagConfig | None = None,
) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    if shared:
        _write_shared_two_region_mesh(mesh_path)
    else:
        _write_disconnected_two_region_mesh(mesh_path)
    simulation = nmag.Simulation(name=name, do_demag=False, config=config)
    simulation.load_mesh(
        str(mesh_path),
        [("first", first), ("second", second)],
        unit_length=nmag.SI(1.0, "m"),
    )
    return simulation


def test_nodal_material_coefficients_follow_disconnected_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _material("first", ms=0.8e6, exchange=1.3e-11, damping=0.1)
    second = _material("second", ms=1.4e6, exchange=2.5e-11, damping=0.3)
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-coefficients",
        first=first,
        second=second,
    )

    coefficients = simulation._nodal_material_coefficients()

    expected_exchange = np.asarray([2.0 * 1.3e-11 / (MU0 * 0.8e6), 2.0 * 2.5e-11 / (MU0 * 1.4e6)])
    np.testing.assert_allclose(coefficients.exchange_prefactor[:4], expected_exchange[0])
    np.testing.assert_allclose(coefficients.exchange_prefactor[4:], expected_exchange[1])
    assert np.all(coefficients.damping[:4] != coefficients.damping[4:])
    assert simulation._nodal_material_coefficients() is coefficients


def test_material_specific_average_only_integrates_selected_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-average",
        first=_material("first"),
        second=_material("second"),
    )
    simulation._fields["m"] = np.asarray(
        [[1.0, 0.0, 0.0]] * 4 + [[0.0, 1.0, 0.0]] * 4,
        dtype=float,
    )

    np.testing.assert_allclose(simulation.get_subfield_average("m", "first"), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(simulation.get_subfield_average("m", "second"), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(simulation.get_subfield_average("m"), [0.5, 0.5, 0.0])

    with pytest.raises(KeyError, match="Unknown material"):
        simulation.get_subfield_average("m", "missing")


def test_heterogeneous_exchange_scales_each_disconnected_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _material("first", ms=1.0e6, exchange=1.0e-11)
    second = _material("second", ms=2.0e6, exchange=4.0e-11)
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-exchange",
        first=first,
        second=second,
    )
    simulation.set_m(
        lambda point: [
            1.0,
            0.2 * (point[0] if point[0] < 5.0 else point[0] - 10.0),
            0.1 * point[1],
        ]
    )

    exchange = np.asarray(simulation.get_subfield("H_exch"), dtype=float)

    first_prefactor = 2.0 * 1.0e-11 / (MU0 * 1.0e6)
    second_prefactor = 2.0 * 4.0e-11 / (MU0 * 2.0e6)
    np.testing.assert_allclose(
        exchange[4:],
        exchange[:4] * (second_prefactor / first_prefactor),
        rtol=2.0e-13,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_heterogeneous_llg_uses_each_regions_coefficients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    if backend == "rust":
        pytest.importorskip("nmag_accel")
    first = _material("first", damping=0.1)
    second = _material("second", damping=0.8)
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name=f"heterogeneous-llg-{backend}",
        first=first,
        second=second,
        config=nmag.NmagConfig(accelerator="rust" if backend == "rust" else "off"),
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    derivative = np.asarray(simulation.get_subfield("dmdt"), dtype=float)

    gamma = 2.210173e5
    expected_first = gamma * 0.1 / (1.0 + 0.1**2) * 1.0e11
    expected_second = gamma * 0.8 / (1.0 + 0.8**2) * 1.0e11
    np.testing.assert_allclose(derivative[:4, 0], expected_first, rtol=2.0e-15)
    np.testing.assert_allclose(derivative[4:, 0], expected_second, rtol=2.0e-15)
    np.testing.assert_allclose(derivative[:, 1:], 0.0, atol=1.0e-12)


def test_heterogeneous_fixed_time_dynamics_separate_material_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _material("first", damping=0.1)
    second = _material("second", damping=0.8)
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-dynamics",
        first=first,
        second=second,
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([1.0e5, 0.0, 0.0], nmag.SI("A/m"))

    simulation.advance_time(nmag.SI(2.0e-12, "s"))
    magnetisation = np.asarray(simulation.get_subfield("m"), dtype=float)

    assert np.all(magnetisation[4:, 0] > magnetisation[:4, 0])
    np.testing.assert_allclose(magnetisation[:4], np.tile(magnetisation[0], (4, 1)))
    np.testing.assert_allclose(magnetisation[4:], np.tile(magnetisation[4], (4, 1)))


def test_conflicting_materials_on_shared_nodes_fail_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-shared",
        first=_material("first", damping=0.1),
        second=_material("second", damping=0.8),
        shared=True,
    )

    with pytest.raises(NotImplementedError, match="material-specific magnetisation DOFs"):
        simulation._nodal_material_coefficients()


def test_identical_material_coefficients_can_share_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="homogeneous-shared",
        first=_material("first", damping=0.25),
        second=_material("second", damping=0.25),
        shared=True,
    )

    coefficients = simulation._nodal_material_coefficients()

    np.testing.assert_array_equal(
        coefficients.damping,
        np.full(5, coefficients.damping[0]),
    )


def test_rust_heterogeneous_llg_matches_python_randomised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("nmag_accel")
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name="heterogeneous-randomised",
        first=_material("first"),
        second=_material("second"),
    )
    generator = np.random.default_rng(20260715)
    m = generator.normal(size=(8, 3))
    m /= np.linalg.norm(m, axis=1)[:, np.newaxis]
    h_total = generator.normal(size=(8, 3)) * 1.0e5
    pin = generator.uniform(0.0, 1.0, size=8)
    scales = generator.uniform(0.5, 1.5, size=8)
    precession = generator.uniform(-3.0e5, 0.0, size=8)
    damping = generator.uniform(-2.0e5, 0.0, size=8)
    normalisation = generator.uniform(0.0, 1.0e11, size=8)

    expected = simulation._llg_rhs_python(
        m,
        h_total,
        pin,
        scales,
        precession,
        damping,
        normalisation,
    )
    actual = simulation._llg_rhs_rust(
        m,
        h_total,
        pin,
        scales,
        precession,
        damping,
        normalisation,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=5.0e-8)


def test_rust_heterogeneous_llg_rejects_mismatched_coefficients() -> None:
    rust_accel = pytest.importorskip("nmag_accel")

    with pytest.raises(ValueError, match="precession_coeff"):
        rust_accel.llg_rhs_heterogeneous(
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 1.0, 0.0]]),
            np.ones(1),
            np.ones(1),
            np.empty(0),
            np.ones(1),
            np.ones(1),
        )
