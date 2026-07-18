from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nmag
import nmesh
from nmag.simulation.support import MU0


def _write_single_tetrahedron(path: Path) -> None:
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        simplices_indices=[[0, 1, 2, 3]],
        simplices_regions=[1],
    )
    mesh.save(path)


def _write_two_regions(path: Path, *, shared: bool) -> None:
    second = (
        [[1.0, 1.0, 1.0]]
        if shared
        else [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]]
    )
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        *second,
    ]
    simplices = [[0, 1, 2, 3], [1, 2, 3, 4]] if shared else [[0, 1, 2, 3], [4, 5, 6, 7]]
    mesh = nmesh.mesh_from_points_and_simplices(
        points=points,
        simplices_indices=simplices,
        simplices_regions=[1, 2],
    )
    mesh.save(path)


def _material(
    name: str,
    anisotropy: object,
    *,
    anisotropy_order: int | None = None,
) -> nmag.MagMaterial:
    return nmag.MagMaterial(
        name=name,
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(0.0, "J/m"),
        anisotropy=anisotropy,  # type: ignore[arg-type]
        anisotropy_order=anisotropy_order,
    )


def _single_simulation(
    tmp_path: Path,
    *,
    anisotropy: object,
    anisotropy_order: int | None = None,
    name: str = "anisotropy",
) -> nmag.Simulation:
    mesh_path = tmp_path / f"{name}.nmesh.h5"
    _write_single_tetrahedron(mesh_path)
    simulation = nmag.Simulation(
        name=name,
        do_demag=False,
        config=nmag.NmagConfig(output_directory=tmp_path),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("magnetic", _material("Py", anisotropy, anisotropy_order=anisotropy_order))],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 1.0])
    return simulation


def test_uniaxial_field_energy_totals_and_output(tmp_path: Path) -> None:
    k1 = 1.0e5
    simulation = _single_simulation(
        tmp_path,
        anisotropy=nmag.uniaxial_anisotropy(
            [0.0, 0.0, 1.0],
            nmag.SI(k1, "J/m^3"),
        ),
    )
    projection = 1.0 / np.sqrt(2.0)
    raw_field = np.asarray([0.0, 0.0, 2.0 * k1 * projection / (MU0 * 1.0e6)])
    m = np.asarray([projection, 0.0, projection])
    expected_field = raw_field - np.dot(raw_field, m) * m

    np.testing.assert_allclose(simulation.get_subfield_average("H_anis"), expected_field)
    assert simulation.get_subfield_average("E_anis") == pytest.approx(-0.5 * k1)
    np.testing.assert_allclose(simulation.get_subfield_average("H_total"), expected_field)
    assert simulation.get_subfield_average("E_total") == pytest.approx(-0.5 * k1)

    simulation.save_data(fields="all")
    header = (tmp_path / "anisotropy_dat.ndt").read_text(encoding="utf-8").splitlines()[1]
    assert "H_anis_Py_2" in header
    assert "E_anis_Py" in header


def test_cubic_and_custom_anisotropy_fields(tmp_path: Path) -> None:
    cubic = _single_simulation(
        tmp_path,
        name="cubic",
        anisotropy=nmag.cubic_anisotropy([1, 0, 0], [0, 1, 0], 1.0e5),
    )
    m = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    gradient = np.asarray(cubic.materials[0].anisotropy.energy_gradient(m))
    raw_field = -gradient / (MU0 * 1.0e6)
    expected_field = raw_field - np.dot(raw_field, m) * m
    np.testing.assert_allclose(
        cubic.get_subfield_average("H_anis"),
        expected_field,
        atol=1.0e-10,
    )

    def custom(m: object) -> nmag.SI:
        return nmag.SI(2.0e5 * np.asarray(m)[0] ** 2, "J/m^3")

    custom_simulation = _single_simulation(
        tmp_path,
        name="custom",
        anisotropy=custom,
        anisotropy_order=2,
    )
    raw_custom = np.asarray([-(4.0e5 / np.sqrt(2.0)) / (MU0 * 1.0e6), 0.0, 0.0])
    expected_custom = raw_custom - np.dot(raw_custom, m) * m
    np.testing.assert_allclose(
        custom_simulation.get_subfield_average("H_anis"),
        expected_custom,
        rtol=1.0e-9,
        atol=1.0e-7,
    )


def test_disconnected_materials_use_their_own_anisotropy(tmp_path: Path) -> None:
    mesh_path = tmp_path / "disconnected.nmesh.h5"
    _write_two_regions(mesh_path, shared=False)
    first = _material("first", nmag.uniaxial_anisotropy([0, 0, 1], 1.0e5))
    second = _material("second", nmag.uniaxial_anisotropy([1, 0, 0], 2.0e5))
    simulation = nmag.Simulation(
        name="disconnected",
        do_demag=False,
        config=nmag.NmagConfig(output_directory=tmp_path),
    )
    simulation.load_mesh(
        str(mesh_path),
        [("first", first), ("second", second)],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 1.0])
    field = np.asarray(simulation.get_subfield("H_anis"))

    assert np.all(field[:4, 0] < 0.0)
    assert np.all(field[:4, 2] > 0.0)
    assert np.all(field[4:, 0] > 0.0)
    assert np.all(field[4:, 2] < 0.0)


def test_shared_nodes_reject_incompatible_anisotropy(tmp_path: Path) -> None:
    mesh_path = tmp_path / "shared.nmesh.h5"
    _write_two_regions(mesh_path, shared=True)
    simulation = nmag.Simulation(
        name="shared",
        do_demag=False,
        config=nmag.NmagConfig(output_directory=tmp_path),
    )
    simulation.load_mesh(
        str(mesh_path),
        [
            ("first", _material("first", nmag.uniaxial_anisotropy([0, 0, 1], 1.0e5))),
            ("second", _material("second", nmag.uniaxial_anisotropy([1, 0, 0], 1.0e5))),
        ],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 1.0])

    with pytest.raises(NotImplementedError, match="material-specific magnetisation DOFs"):
        simulation.get_subfield("H_anis")


def test_shared_nodes_accept_equivalent_predefined_anisotropy(tmp_path: Path) -> None:
    mesh_path = tmp_path / "shared-equivalent.nmesh.h5"
    _write_two_regions(mesh_path, shared=True)
    simulation = nmag.Simulation(
        name="shared-equivalent",
        do_demag=False,
        config=nmag.NmagConfig(output_directory=tmp_path),
    )
    simulation.load_mesh(
        str(mesh_path),
        [
            ("first", _material("first", nmag.uniaxial_anisotropy([0, 0, 1], 1.0e5))),
            ("second", _material("second", nmag.uniaxial_anisotropy([0, 0, 1], 1.0e5))),
        ],
        unit_length=nmag.SI(1.0e-9, "m"),
    )
    simulation.set_m([1.0, 0.0, 1.0])

    assert np.isfinite(simulation.get_subfield("H_anis")).all()
