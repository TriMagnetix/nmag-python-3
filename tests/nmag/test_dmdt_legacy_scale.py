from __future__ import annotations

import numpy as np
import pytest

import nmag
import nmesh


def _write_tetrahedral_mesh(path: str) -> None:
    nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    ).save(path)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_dmdt_uses_legacy_global_magnetisation_scale(
    tmp_path,
    monkeypatch,
    backend: str,
) -> None:
    if backend == "rust":
        pytest.importorskip("nmag_accel")

    mesh_path = tmp_path / "tetra.nmesh.h5"
    _write_tetrahedral_mesh(str(mesh_path))
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(0.86e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
        do_precession=False,
    )
    simulation = nmag.Simulation(
        name="legacy-dmdt",
        do_demag=False,
        config=nmag.NmagConfig(accelerator="rust" if backend == "rust" else "off"),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", material)],
        unit_length=nmag.SI(1.0, "m"),
    )
    simulation.set_m([0.0, 1.0, 0.0])
    simulation.set_H_ext([2.0, 0.0, 3.0], nmag.SI("A/m"))

    damping_coeff = -221017.3 * 0.5 / (1.0 + 0.5**2)
    expected = -damping_coeff * 1.0e6 * np.asarray([2.0, 0.0, 3.0])
    dmdt = simulation._subfield_array("dmdt")

    np.testing.assert_allclose(dmdt, np.tile(expected, (4, 1)))
