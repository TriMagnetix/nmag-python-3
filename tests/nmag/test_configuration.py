from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import public_api_support_module as helpers
import pytest

import nmag
import nmag.backends as backends
from nmag.backends import _selected_llg_backend, _selected_maxangle_backend


def _simulation(tmp_path, *, config: nmag.NmagConfig, name: str = "configured") -> nmag.Simulation:
    mesh_path = tmp_path / "mesh.nmesh"
    helpers.write_single_region_mesh(mesh_path)
    material = nmag.MagMaterial(
        name="Py",
        Ms=nmag.SI(1.0e6, "A/m"),
        exchange_coupling=nmag.SI(13.0e-12, "J/m"),
    )
    simulation = nmag.Simulation(name=name, config=config)
    simulation.load_mesh(str(mesh_path), [("generated", material)], unit_length=nmag.SI(1e-9, "m"))
    simulation.set_m([1.0, 0.0, 0.0])
    return simulation


def test_config_defaults_are_immutable_and_overrides_are_frozen() -> None:
    config = nmag.NmagConfig(accelerator_overrides={nmag.RustKernel.LLG: "off"})

    assert config.default_name == "nmag_simulation"
    assert config.output_policy == "error"
    assert config.demag_bem_storage == "auto"
    assert config.hierarchical_bem.relative_tolerance == 1.0e-6
    assert config.accelerator_mode_for(nmag.RustKernel.LLG) == "off"
    with pytest.raises(FrozenInstanceError):
        config.accelerator = "off"  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.accelerator_overrides[nmag.RustKernel.MAXANGLE] = "off"  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_policy": "discard"}, "output_policy"),
        ({"accelerator": "gpu"}, "accelerator"),
        ({"integrator_backend": "cvode"}, "integrator_backend"),
        ({"demag_bem_storage": "compressed"}, "demag_bem_storage"),
        ({"accelerator_overrides": {"llg": "off"}}, "keys"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nmag.NmagConfig(**kwargs)


def test_explicit_config_wins_over_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(nmag.ACCELERATOR_ENV, "off")
    monkeypatch.setenv("NMAG_DEMAG_BEM_STORAGE_BACKEND", "matrix-free")
    explicit = nmag.NmagConfig(
        accelerator="rust",
        output_directory=tmp_path,
        demag_bem_storage="hierarchical",
    )
    simulation = nmag.Simulation(config=explicit)

    assert simulation.config is explicit
    assert simulation.config.accelerator == "rust"
    assert simulation.config.demag_bem_storage == "hierarchical"
    assert nmag.Simulation(config=None).config.accelerator == "off"
    assert nmag.Simulation(config=None).config.demag_bem_storage == "matrix-free"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"relative_tolerance": 0.0}, "relative_tolerance"),
        ({"admissibility_eta": float("inf")}, "admissibility_eta"),
        ({"leaf_size": 0}, "leaf_size"),
        ({"max_rank": True}, "max_rank"),
        ({"memory_fraction": 1.1}, "memory_fraction"),
    ],
)
def test_hierarchical_bem_config_rejects_invalid_values(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nmag.HierarchicalBemConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relative_tolerance": "1e-6"},
        {"admissibility_eta": True},
        {"memory_fraction": "0.2"},
    ],
)
def test_hierarchical_bem_config_rejects_non_numeric_values(kwargs) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        nmag.HierarchicalBemConfig(**kwargs)


def test_config_requires_hierarchical_bem_config() -> None:
    with pytest.raises(TypeError, match="HierarchicalBemConfig"):
        nmag.NmagConfig(hierarchical_bem=None)  # type: ignore[arg-type]


def test_off_auto_rust_and_kernel_override_selection(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_rust_accelerator_available", lambda: True)
    assert _selected_llg_backend(10_000, nmag.NmagConfig(accelerator="off")) == "python"
    assert _selected_llg_backend(10_000, nmag.NmagConfig()) == "rust"
    assert _selected_llg_backend(1, nmag.NmagConfig(accelerator="rust")) == "rust"
    assert _selected_maxangle_backend(
        nmag.NmagConfig(
            accelerator="rust",
            accelerator_overrides={nmag.RustKernel.MAXANGLE: "off"},
        )
    ) == "python"


def test_strict_rust_selection_fails_clearly_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_rust_accelerator_available", lambda: False)
    monkeypatch.setattr(
        backends,
        "_load_rust_accelerator",
        lambda _required_by: (_ for _ in ()).throw(RuntimeError("extension missing")),
    )

    with pytest.raises(RuntimeError, match="extension missing"):
        _selected_llg_backend(1, nmag.NmagConfig(accelerator="rust"))


def test_output_policy_error_replace_and_append(tmp_path) -> None:
    error_config = nmag.NmagConfig(output_directory=tmp_path)
    first = _simulation(tmp_path, config=error_config)
    first.save_data()
    ndt = tmp_path / "configured_dat.ndt"
    original = ndt.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="output_policy"):
        nmag.Simulation(name="configured", config=error_config)

    replacement = _simulation(
        tmp_path,
        config=nmag.NmagConfig(output_directory=tmp_path, output_policy="replace"),
    )
    replacement.save_data()
    assert ndt.read_text(encoding="utf-8").count("# Simulation:") == 1

    appended = _simulation(
        tmp_path,
        config=nmag.NmagConfig(output_directory=tmp_path, output_policy="append"),
    )
    appended.clock.step = 1
    appended.save_data()
    rows = [line for line in ndt.read_text(encoding="utf-8").splitlines() if line]
    assert rows[0].startswith("# Simulation:")
    assert len(rows) == 4
    assert original.splitlines()[1] == rows[1]


def test_append_rejects_a_different_schema_without_writing(tmp_path) -> None:
    config = nmag.NmagConfig(output_directory=tmp_path)
    first = _simulation(tmp_path, config=config)
    first.save_data()
    ndt = tmp_path / "configured_dat.ndt"
    before = ndt.read_text(encoding="utf-8")

    appended = _simulation(
        tmp_path,
        config=nmag.NmagConfig(output_directory=tmp_path, output_policy="append"),
    )
    appended.writer._column_names = ["wrong"]
    appended.writer._column_index_by_name = {"wrong": 0}
    appended.writer._append_schema_pending = False
    with pytest.raises(ValueError, match="different NDT schema"):
        appended.save_data()
    assert ndt.read_text(encoding="utf-8") == before


def test_explicit_restart_is_independent_from_output_append(tmp_path) -> None:
    simulation = _simulation(tmp_path, config=nmag.NmagConfig(output_directory=tmp_path))
    checkpoint = simulation.save_restart_file()
    assert checkpoint.exists()
    resumed = _simulation(
        tmp_path,
        config=nmag.NmagConfig(output_directory=tmp_path, output_policy="append"),
    )
    assert np.asarray(resumed.get_subfield("m")).shape == (4, 3)
    resumed.load_restart_file(checkpoint)
