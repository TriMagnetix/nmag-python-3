from __future__ import annotations

import json
import subprocess
import sys


def _run_module_probe(source: str, *args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_importing_nmag_does_not_import_mesher_stack():
    result = _run_module_probe(
        """
import json
import sys
import nmag

print(json.dumps({
    "has_nmag": hasattr(nmag, "Simulation"),
    "nmesh_loaded": "nmesh" in sys.modules,
    "geometry_loaded": "nmesh.geometry" in sys.modules,
    "mesher_loaded": "nmesh.mesher" in sys.modules,
    "relaxation_loaded": "nmesh.mesher.relaxation" in sys.modules,
    "meshio_loaded": "meshio" in sys.modules,
    "numba_loaded": "numba" in sys.modules,
}))
"""
    )

    assert result == {
        "has_nmag": True,
        "nmesh_loaded": True,
        "geometry_loaded": False,
        "mesher_loaded": False,
        "relaxation_loaded": False,
        "meshio_loaded": False,
        "numba_loaded": False,
    }


def test_nmesh_lazy_exports_preserve_public_geometry_and_mesher_symbols():
    result = _run_module_probe(
        """
import json
import sys
import nmesh

before = {
    "geometry_loaded": "nmesh.geometry" in sys.modules,
    "mesher_loaded": "nmesh.mesher" in sys.modules,
    "relaxation_loaded": "nmesh.mesher.relaxation" in sys.modules,
}
from nmesh.nmesh import Box as DirectBox
box_name = nmesh.Box.__name__
direct_box_name = DirectBox.__name__
params_name = nmesh.get_default_meshing_parameters().__class__.__name__
after = {
    "geometry_loaded": "nmesh.geometry" in sys.modules,
    "mesher_loaded": "nmesh.mesher" in sys.modules,
    "relaxation_loaded": "nmesh.mesher.relaxation" in sys.modules,
}
print(json.dumps({
    "before": before,
    "box_name": box_name,
    "direct_box_name": direct_box_name,
    "params_name": params_name,
    "after": after,
}))
"""
    )

    assert result["before"] == {
        "geometry_loaded": False,
        "mesher_loaded": False,
        "relaxation_loaded": False,
    }
    assert result["box_name"] == "Box"
    assert result["direct_box_name"] == "Box"
    assert result["params_name"] == "MeshingParameters"
    assert result["after"] == {
        "geometry_loaded": True,
        "mesher_loaded": True,
        "relaxation_loaded": False,
    }


def test_nmesh_load_legacy_hdf5_does_not_import_meshio(tmp_path):
    result = _run_module_probe(
        """
import json
from pathlib import Path
import sys

import nmesh
from nmesh.backend import RawMesh
from nmesh.io.legacy_nmesh_hdf5 import save_raw_mesh_as_legacy_nmesh_hdf5

path = Path(sys.argv[1])
save_raw_mesh_as_legacy_nmesh_hdf5(
    path,
    RawMesh(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        simplices=[[0, 1, 2, 3]],
        regions=[1],
        dim=3,
    ),
)
loaded = nmesh.load(path)
print(json.dumps({
    "meshio_loaded": "meshio" in sys.modules,
    "point_count": len(loaded.points),
    "simplex_count": len(loaded.simplices),
}))
""",
        str(tmp_path / "direct.nmesh.h5"),
    )

    assert result == {
        "meshio_loaded": False,
        "point_count": 4,
        "simplex_count": 1,
    }
