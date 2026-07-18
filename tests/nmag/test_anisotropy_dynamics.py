from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

import nmag
import nmesh


def _simulation(
    tmp_path: Path,
    *,
    name: str,
    k1: float,
    accelerator: Literal["auto", "off", "rust"] = "off",
) -> nmag.Simulation:
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(mesh_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(0.0, "J/m"),
        llg_damping=0.5,
        do_precession=False,
        anisotropy=nmag.uniaxial_anisotropy([0.0, 0.0, 1.0], nmag.SI(k1, "J/m^3")),
    )
    simulation = nmag.Simulation(
        name=name,
        do_demag=False,
        config=nmag.NmagConfig(
            output_directory=tmp_path,
            accelerator=accelerator,
        ),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 1.0])
    return simulation


@pytest.mark.parametrize(("k1", "direction"), [(1.0e5, 1.0), (-1.0e5, -1.0)])
def test_anisotropy_damping_moves_toward_lower_energy(
    tmp_path: Path,
    k1: float,
    direction: float,
) -> None:
    simulation = _simulation(tmp_path, name=f"axis-{direction}", k1=k1)
    initial = np.asarray(simulation.get_subfield_average("m"))
    initial_energy = float(simulation.get_subfield_average("E_anis"))

    simulation.advance_time(nmag.SI(1.0e-12, "s"))

    final = np.asarray(simulation.get_subfield_average("m"))
    final_energy = float(simulation.get_subfield_average("E_anis"))
    assert direction * (final[2] - initial[2]) > 0.0
    assert final_energy < initial_energy
    np.testing.assert_allclose(np.linalg.norm(final), 1.0, rtol=2.0e-6)


def test_python_and_rust_anisotropy_dynamics_match(tmp_path: Path) -> None:
    pytest.importorskip("nmag_accel")
    python = _simulation(tmp_path, name="python-anis", k1=1.0e5, accelerator="off")
    rust = _simulation(tmp_path, name="rust-anis", k1=1.0e5, accelerator="rust")

    target = nmag.SI(2.0e-13, "s")
    python.advance_time(target)
    rust.advance_time(target)

    np.testing.assert_allclose(
        rust.get_subfield("m"),
        python.get_subfield("m"),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
