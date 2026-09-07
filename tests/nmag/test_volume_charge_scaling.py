from __future__ import annotations

from pathlib import Path

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
from nmag.simulation.demag.fem.charges import volume_charge_scaling_correction


def _simulation(
    tmp_path: Path,
    *,
    name: str,
    scale: float,
    accelerator: str = "off",
) -> nmag.Simulation:
    mesh_path = tmp_path / "mesh.nmesh"
    if not mesh_path.exists():
        helpers.write_two_tetra_mesh(mesh_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        scale_volume_charges=scale,
    )
    simulation = nmag.Simulation(
        name=name,
        config=nmag.NmagConfig(accelerator=accelerator),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("generated", material)],
        unit_length=nmag.SI(1e-9, "m"),
    )
    return simulation


def _nonuniform_m(position: np.ndarray) -> list[float]:
    return [1.0 + position[0] * 1e9, 0.25 + position[1] * 1e9, 0.5]


def test_volume_charge_correction_matches_linear_tetrahedron_integral():
    simplices = np.asarray([[0, 1, 2, 3]], dtype=int)
    gradients = np.asarray(
        [[[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    volumes = np.asarray([1.0 / 6.0])
    m = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    correction = volume_charge_scaling_correction(
        simplices,
        gradients,
        volumes,
        m,
        np.asarray([2.0]),
        np.asarray([0.25]),
        point_count=4,
    )

    # div(M) is 2.0, and each local linear basis function integrates to V/4.
    np.testing.assert_allclose(correction, np.full(4, 0.0625))


@pytest.mark.parametrize("scale", [0.0, 0.25, 2.0])
def test_uniform_magnetisation_has_no_volume_charge_effect(tmp_path, monkeypatch, scale):
    monkeypatch.chdir(tmp_path)
    reference = _simulation(tmp_path, name="reference", scale=1.0)
    scaled = _simulation(tmp_path, name="scaled", scale=scale)
    reference.set_m([1.0, 0.0, 0.0])
    scaled.set_m([1.0, 0.0, 0.0])

    np.testing.assert_allclose(
        scaled.get_subfield("H_demag"),
        reference.get_subfield("H_demag"),
        rtol=1e-12,
        atol=1e-9,
    )


def test_nonuniform_magnetisation_responds_linearly_to_volume_charge_scale(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    simulations = [
        _simulation(tmp_path, name=f"scale-{scale}", scale=scale) for scale in (0.0, 1.0, 2.0)
    ]
    for simulation in simulations:
        simulation.set_m(_nonuniform_m)

    fields = [np.asarray(simulation.get_subfield("H_demag")) for simulation in simulations]
    assert not np.allclose(fields[0], fields[1])
    np.testing.assert_allclose(fields[2] - fields[1], fields[1] - fields[0], rtol=1e-10, atol=1e-7)


def test_scaled_volume_charge_matches_between_python_and_rust(tmp_path, monkeypatch):
    pytest.importorskip("nmag_accel")
    monkeypatch.chdir(tmp_path)
    python_simulation = _simulation(tmp_path, name="python", scale=0.25, accelerator="off")
    rust_simulation = _simulation(tmp_path, name="rust", scale=0.25, accelerator="rust")
    python_simulation.set_m(_nonuniform_m)
    rust_simulation.set_m(_nonuniform_m)

    np.testing.assert_allclose(
        rust_simulation.get_subfield("H_demag"),
        python_simulation.get_subfield("H_demag"),
        rtol=1e-10,
        atol=1e-7,
    )


def test_scaled_volume_charge_matches_legacy_two_tetra_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = _simulation(tmp_path, name="legacy-parity", scale=2.0)
    simulation.set_m(_nonuniform_m)

    expected_legacy_h = np.asarray(
        [
            [-100428.36054294, -58805.94740934, -53098.23403883],
            [-73932.89295219, -32310.47981859, -26602.76644807],
            [-73932.89295219, -32310.47981859, -26602.76644807],
            [-73932.89295219, -32310.47981859, -26602.76644807],
            [-60685.15915681, -19062.74602321, -13355.0326527],
        ]
    )
    expected_legacy_rho = np.asarray(
        [-2.76517975e14, 7.54482569e13, -1.46288202e14, -1.76695814e14, 1.95556183e14]
    )

    np.testing.assert_allclose(
        simulation.get_subfield("H_demag"),
        expected_legacy_h,
        # Legacy's iterative potential solve adds source-dependent roundoff;
        # rho below is the tighter parity check for the scaled operator.
        rtol=4e-5,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        simulation.get_subfield("rho"),
        expected_legacy_rho,
        rtol=1e-8,
        atol=1e5,
    )


def test_volume_charge_scales_follow_mesh_regions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_region_mesh(mesh_path)
    materials = [
        nmag.MagMaterial(name="A", scale_volume_charges=0.25),
        nmag.MagMaterial(name="B", scale_volume_charges=2.0),
    ]
    simulation = nmag.Simulation(name="regions")
    simulation.load_mesh(
        str(mesh_path),
        [("region-a", materials[0]), ("region-b", materials[1])],
        unit_length=nmag.SI(1e-9, "m"),
    )

    np.testing.assert_array_equal(
        simulation._simplex_volume_charge_scales([1, 2]),
        [0.25, 2.0],
    )


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf")])
def test_volume_charge_scale_must_be_finite(scale):
    with pytest.raises(ValueError, match="scale_volume_charges must be finite"):
        nmag.MagMaterial(name="invalid", scale_volume_charges=scale)


def test_volume_charge_scale_must_be_numeric():
    with pytest.raises(TypeError, match="scale_volume_charges must be a finite real number"):
        nmag.MagMaterial(name="invalid", scale_volume_charges="invalid")  # type: ignore[arg-type]


def test_restart_rejects_different_volume_charge_scale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = _simulation(tmp_path, name="source", scale=0.25)
    source.set_m(_nonuniform_m)
    checkpoint = source.save_restart_file(tmp_path / "scaled-restart.h5")

    target = _simulation(tmp_path, name="target", scale=1.0)
    target.set_m([0.0, 1.0, 0.0])
    original_m = np.array(target.get_subfield("m"), copy=True)

    with pytest.raises(ValueError, match="materials do not match"):
        target.load_restart_file(checkpoint)
    np.testing.assert_array_equal(target.get_subfield("m"), original_m)
