from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import public_api_support_module as helpers
import pytest

import nmag

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
SPHERE1_MESH = FIXTURES_DIR / "nmag_doc_example1" / "sphere1.nmesh.h5"



def test_simulation_mvp_load_set_save_and_probe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)

    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )

    sim = nmag.Simulation(name="mvp")
    sim.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    assert sim.get_subfield_average("H_anis") is None
    assert "constant_average" not in sim.last_subfield_average_timings_seconds

    sim.set_m([1.0, 0.0, 0.0])
    sim.set_H_ext([1.0, 2.0, 3.0], nmag.SI("A/m"))
    sim.save_data(fields="all")

    assert sim.probe_subfield_siv("H_ext", [0.0, 0.0, 0.0]) == [1.0, 2.0, 3.0]
    assert sim.probe_subfield_siv("m", [0.0, 0.0, 0.0]) == [1.0, 0.0, 0.0]

    with h5py.File(tmp_path / "mvp_dat.h5") as h5:
        assert set(h5["fields"].keys()) == {
            "E_anis",
            "E_demag",
            "E_exch",
            "E_ext",
            "E_total",
            "H_anis",
            "H_demag",
            "H_ext",
            "H_exch",
            "H_total",
            "M",
            "dmdt",
            "m",
            "pin",
            "phi",
            "rho",
        }
        assert h5["mesh/points"].shape == (4, 3)
        assert h5["fields/m"].shape == (4, 3)
        assert h5["fields/M"].shape == (4, 3)
        assert h5["fields/dmdt"].shape == (4, 3)
        assert h5["fields/H_ext"].shape == (4, 3)
        assert h5["fields/H_total"].shape == (4, 3)
        np.testing.assert_allclose(
            h5["fields/H_ext"][:],
            np.tile([1.0, 2.0, 3.0], (4, 1)),
        )
        assert h5["fields/H_demag"].shape == (4, 3)
        assert h5["fields/E_demag"].shape == (4,)
        assert h5["fields/pin"].shape == (4,)
        assert h5["fields/phi"].shape == (4,)
        assert h5["fields/rho"].shape == (4,)

    assert (tmp_path / "mvp_dat.ndt").exists()
    header = (tmp_path / "mvp_dat.ndt").read_text(encoding="utf-8").splitlines()[1].split("\t")
    for column in [
        "last_step_dt",
        "M_Py_0",
        "H_total_Py_0",
        "H_anis_Py_0",
        "H_exch_Py_0",
        "E_total_Py",
        "E_ext_Py",
        "E_demag_Py",
        "pin",
        "phi",
        "rho",
        "maxangle_m_Py",
    ]:
        assert column in header


def test_simulation_rejects_unimplemented_constructor_options():
    with pytest.raises(NotImplementedError, match="Periodic boundary"):
        nmag.Simulation(periodic_bc=[1, 0, 0])

    with pytest.raises(NotImplementedError, match="Spin-transfer torque"):
        nmag.Simulation(do_sl_stt=True)


def test_demag_disabled_does_not_advertise_or_calculate_demag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)
    material = nmag.MagMaterial(name="Py", Ms=nmag.SI(1.0e6, "A/m"))
    sim = nmag.Simulation(name="no-demag", do_demag=False)
    sim.load_mesh(
        str(mesh_path),
        [("generated", material)],
        unit_length=nmag.SI(1e-9, "m"),
    )
    sim.set_m([1.0, 0.0, 0.0])

    assert not sim.is_subfield_available("H_demag")
    assert not sim.is_subfield_available("phi")
    assert not sim.is_subfield_available("rho")
    assert not sim.is_subfield_available("E_demag")
    with pytest.raises(KeyError, match="demag is disabled"):
        sim.get_subfield("H_demag")


def test_load_mesh_rejects_missing_material_region(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_two_region_mesh(mesh_path)
    material = nmag.MagMaterial(name="Py", Ms=nmag.SI(1.0e6, "A/m"))
    sim = nmag.Simulation(name="missing-region")

    with pytest.raises(ValueError, match=r"mesh=\[1, 2\], configured=\[1\]"):
        sim.load_mesh(
            str(mesh_path),
            [("region-one", material)],
            unit_length=nmag.SI(1e-9, "m"),
        )


