from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import nmag
import nmesh
from si.physical import SI


def _simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    polarisation: float = 0.7,
    xi: float = 0.02,
    damping: float = 0.1,
    config: nmag.NmagConfig | None = None,
) -> nmag.Simulation:
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(str(mesh_path))
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(8.0e5, "A/m"),
        exchange_coupling=nmag.SI(0.0, "J/m"),
        llg_damping=damping,
        llg_polarisation=polarisation,
        llg_xi=xi,
        do_precession=False,
    )
    simulation = nmag.Simulation(name=name, do_demag=False, config=config)
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0, "m"),
    )
    simulation.set_m([1.0, 0.0, 0.0])
    return simulation


def test_current_density_accepts_uniform_nodal_and_callable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-inputs")

    simulation.set_current_density([1.0, 2.0, 3.0], nmag.SI("A/m^2"))
    np.testing.assert_array_equal(
        simulation.get_subfield("current_density"),
        np.tile([1.0, 2.0, 3.0], (4, 1)),
    )

    nodal = np.asarray(
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
    )
    simulation.set_current_density(nodal, nmag.SI("A/m^2"))
    np.testing.assert_array_equal(simulation.get_subfield("current_density"), nodal)

    simulation.set_current_density(
        lambda point: [point[0], point[1], point[2]],
        nmag.SI(2.0, "A/m^2"),
    )
    assert simulation.mesh is not None
    np.testing.assert_array_equal(
        simulation.get_subfield("current_density"),
        2.0 * np.asarray(simulation.mesh.points),
    )


@pytest.mark.parametrize(
    "values",
    ([1.0, 2.0], np.ones((4, 2)), [1.0, 2.0, float("nan")]),
)
def test_current_density_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    values: object,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-invalid")

    with pytest.raises(ValueError, match="current_density"):
        simulation.set_current_density(values, nmag.SI("A/m^2"))  # type: ignore[arg-type]


def test_directional_derivative_recovers_linear_tetrahedral_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-gradient")
    assert simulation.mesh is not None
    simulation._fields["m"] = np.asarray(simulation.mesh.points, dtype=float)
    simulation.set_current_density([2.0, -3.0, 5.0], nmag.SI("A/m^2"))

    np.testing.assert_allclose(
        simulation.get_subfield("dm_dcurrent"),
        np.tile([2.0, -3.0, 5.0], (4, 1)),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_uniform_magnetisation_has_zero_current_torque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-uniform")
    simulation.set_current_density([0.0, 0.0, 1.0e12], nmag.SI("A/m^2"))

    np.testing.assert_allclose(simulation.get_subfield("dm_dcurrent"), 0.0, atol=0.0)
    np.testing.assert_allclose(simulation.get_subfield("dmdt"), 0.0, atol=0.0)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_current_torque_evolves_nonuniform_state_through_integrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    if backend == "rust":
        pytest.importorskip("nmag_accel")
    simulation = _simulation(
        tmp_path,
        monkeypatch,
        name=f"stt-dynamics-{backend}",
        config=nmag.NmagConfig(accelerator="rust" if backend == "rust" else "off"),
    )
    simulation.set_m(lambda point: [1.0, 0.1 * point[0], 0.0])
    simulation.set_current_density([1.0e12, 0.0, 0.0], nmag.SI("A/m^2"))
    initial = np.asarray(simulation.get_subfield("m"), dtype=float)

    simulation.advance_time(nmag.SI(1.0e-12, "s"))

    final = np.asarray(simulation.get_subfield("m"), dtype=float)
    assert np.max(np.abs(final - initial)) > 1.0e-13
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), 1.0, atol=1.0e-12)


def test_python_stt_rhs_matches_legacy_formula_and_pinning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-formula")
    m = np.asarray([[0.6, 0.8, 0.0], [1.0, 0.0, 0.0]])
    directional = np.asarray([[2.0, -1.0, 3.0], [4.0, 5.0, 6.0]])
    pin = np.asarray([1.0, 0.0])
    c4 = np.asarray([-2.0, -3.0])
    c5 = np.asarray([0.25, 0.5])

    actual = simulation._llg_rhs_python(
        m,
        np.zeros_like(m),
        pin,
        np.ones(2),
        0.0,
        0.0,
        0.0,
        directional,
        c4,
        c5,
    )
    mdotg = np.einsum("ij,ij->i", m, directional)
    mdotm = np.einsum("ij,ij->i", m, m)
    expected = (
        c4[:, None] * (m * mdotg[:, None] - directional * mdotm[:, None])
        + c5[:, None] * np.cross(m, directional)
    ) * pin[:, None]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(actual[1], np.zeros(3))


def test_material_stt_coefficients_match_legacy_definition() -> None:
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(8.0e5, "A/m"),
        llg_damping=0.1,
        llg_polarisation=0.7,
        llg_xi=0.02,
    )
    from si import constants

    gilbert = 1.0 / (1.0 + 0.1**2)
    bohr_magneton = (1.0 * constants.bohr_magneton).to("J/T").magnitude
    charge = (1.0 * constants.positron_charge).to("C").magnitude
    prefactor = -gilbert * 0.7 * bohr_magneton / (charge * 8.0e5 * (1.0 + 0.02**2))

    adiabatic = cast(SI, material.su_llg_stt_adiab)
    nonadiabatic = cast(SI, material.su_llg_stt_nadiab)
    assert adiabatic.in_units_of(nmag.SI("m^3/A/s")) == pytest.approx(
        prefactor * (1.0 + 0.1 * 0.02)
    )
    assert nonadiabatic.in_units_of(nmag.SI("m^3/A/s")) == pytest.approx(
        prefactor * (0.02 - 0.1)
    )


def test_python_and_rust_stt_rhs_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("nmag_accel")
    simulation = _simulation(tmp_path, monkeypatch, name="stt-rust")
    generator = np.random.default_rng(20260715)
    m = generator.normal(size=(16, 3))
    m /= np.linalg.norm(m, axis=1)[:, None]
    h_total = generator.normal(size=(16, 3)) * 1.0e5
    directional = generator.normal(size=(16, 3)) * 1.0e12
    pin = generator.uniform(0.0, 1.0, size=16)
    scales = generator.uniform(0.5, 1.5, size=16)
    precession = generator.uniform(-3.0e5, 0.0, size=16)
    damping = generator.uniform(-2.0e5, 0.0, size=16)
    normalisation = generator.uniform(0.0, 1.0e11, size=16)
    c4 = generator.uniform(-1.0e-10, 0.0, size=16)
    c5 = generator.uniform(-1.0e-10, 1.0e-10, size=16)

    expected = simulation._llg_rhs_python(
        m,
        h_total,
        pin,
        scales,
        precession,
        damping,
        normalisation,
        directional,
        c4,
        c5,
    )
    actual = simulation._llg_rhs_rust(
        m,
        h_total,
        pin,
        scales,
        precession,
        damping,
        normalisation,
        directional,
        c4,
        c5,
    )

    np.testing.assert_allclose(actual, expected, rtol=5.0e-15, atol=5.0e-8)


def test_rust_stt_rhs_rejects_invalid_directional_derivative() -> None:
    rust_accel = pytest.importorskip("nmag_accel")

    with pytest.raises(ValueError, match="dm_dcurrent"):
        rust_accel.llg_rhs_stt_heterogeneous(
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 1.0, 0.0]]),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.empty((0, 3)),
            np.ones(1),
            np.ones(1),
        )


def test_current_density_invalidates_initialised_integrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = _simulation(tmp_path, monkeypatch, name="stt-invalidate")
    simulation.reinitialise()
    assert simulation._integrator_is_stale is False

    simulation.set_current_density([0.0, 0.0, 1.0], nmag.SI("A/m^2"))

    assert simulation._integrator_is_stale is True
