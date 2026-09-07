from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import nmag
import nmesh
from nmag.simulation import (
    TETRA_FACE_VERTICES,
    _si_dimensionless,
    _si_unit,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"

def test_simulation_unit_helpers_cache_common_si_units():
    _si_unit.cache_clear()
    _si_dimensionless.cache_clear()

    assert _si_unit("A/m") is _si_unit("A/m")
    assert _si_unit("A/m").in_units_of(nmag.SI(1, "A/m")) == 1.0
    assert _si_dimensionless() is _si_dimensionless()
    assert _si_dimensionless().in_units_of(nmag.SI(1)) == 1.0


def _subprocess_env_with_repo_src():
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    pythonpath = os.environ.get("PYTHONPATH")
    return {
        **os.environ,
        "PYTHONPATH": (
            src_path
            if not pythonpath
            else f"{src_path}{os.pathsep}{pythonpath}"
        ),
    }


def _write_single_region_mesh(path):
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
    mesh.save(path)


def _write_two_tetra_mesh(path):
    mesh = nmesh.mesh_from_points_and_simplices(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        simplices_indices=[
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        simplices_regions=[1, 1],
    )
    mesh.save(path)


def _write_two_region_mesh(path):
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
    mesh.save(path)


def _reference_oriented_boundary_faces(points, simplices):
    face_owners = {}
    for cell_index, simplex in enumerate(simplices):
        cell_center = np.mean(points[simplex], axis=0)
        for local_face in TETRA_FACE_VERTICES:
            face = tuple(int(simplex[index]) for index in local_face)
            key = tuple(sorted(face))
            if key in face_owners:
                face_owners[key] = None
                continue

            a, b, c = points[list(face)]
            normal = np.cross(b - a, c - a)
            face_center = (a + b + c) / 3.0
            if np.dot(normal, cell_center - face_center) > 0.0:
                face = (face[0], face[2], face[1])
            face_owners[key] = (cell_index, face)

    return [owner for owner in face_owners.values() if owner is not None]


def test_public_nmag_import_surface():
    assert nmag.SI(1, "A/m").dens_str() == "<A/m>"
    assert nmag.IntegratorConfig
    assert nmag.IntegratorStats
    assert nmag.MagMaterial
    assert nmag.Simulation


def test_nmag_simulation_import_keeps_scipy_linalg_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import nmag.simulation as sim; "
                "print(sim._SCIPY_LINALG is None); "
                "print('scipy.linalg' in sys.modules); "
                "print('simulation.quantity' in sys.modules); "
                "print('simulation.data_writer' in sys.modules)"
            ),
        ],
        check=True,
        env=_subprocess_env_with_repo_src(),
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["True", "False", "False", "False"]


def test_public_nmag_import_keeps_material_constants_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import nmag; "
                "print('si.constants' in sys.modules); "
                "print('simulation.quantity' in sys.modules); "
                "print('simulation.data_writer' in sys.modules); "
                "material = nmag.MagMaterial(name='Py'); "
                "print('si.constants' in sys.modules); "
                "_ = material.thermal_factor; "
                "print('si.constants' in sys.modules)"
            ),
        ],
        check=True,
        env=_subprocess_env_with_repo_src(),
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["False", "False", "False", "False", "True"]


def test_public_nmag_import_keeps_pint_registry_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import nmag; "
                "print('pint' in sys.modules); "
                "print(nmag.SI.__name__); "
                "print('pint' in sys.modules); "
                "sim = nmag.Simulation(name='probe'); "
                "mat = nmag.MagMaterial("
                "name='Py', "
                "Ms=nmag.SI(1e6, 'A/m'), "
                "exchange_coupling=nmag.SI(13e-12, 'J/m')); "
                "_ = nmag.SI(1, 'm'); "
                "_ = nmag.SI(1e6, 'A/m'); "
                "_ = nmag.SI(13e-12, 'J/m'); "
                "_ = nmag.SI(1, 's') / nmag.SI(1e-12, 's'); "
                "_ = mat.Ms.in_units_of(nmag.SI(1, 'A/m')); "
                "print('pint' in sys.modules); "
                "_ = nmag.SI(1, 'Wb'); "
                "print('pint' in sys.modules)"
            ),
        ],
        check=True,
        env=_subprocess_env_with_repo_src(),
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "False",
        "Physical",
        "False",
        "False",
        "True",
    ]


