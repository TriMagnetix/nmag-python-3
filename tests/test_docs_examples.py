from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_EXAMPLES = REPOSITORY_ROOT / "docs" / "examples"
DOCS_ASSETS = REPOSITORY_ROOT / "docs" / "assets"
FIXTURE_MESH = REPOSITORY_ROOT / "tests" / "fixtures" / "nmag_doc_example1" / "sphere1.nmesh.h5"


def _run_example(script: Path, working_directory: Path, *arguments: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(argument) for argument in arguments)],
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )


def test_quickstart_example_runs_and_writes_results(tmp_path: Path) -> None:
    completed = _run_example(DOCS_EXAMPLES / "quickstart.py", tmp_path)

    assert "Average demagnetization field:" in completed.stdout
    assert (tmp_path / "results" / "quickstart.nmesh.h5").is_file()
    assert (tmp_path / "results" / "quickstart_dat.ndt").is_file()
    with h5py.File(tmp_path / "results" / "quickstart_dat.h5") as output:
        assert output["fields/m"].shape == (4, 3)
        assert output["fields/H_demag"].shape == (4, 3)


def test_sphere_download_matches_fixture_and_example_runs(tmp_path: Path) -> None:
    downloadable_mesh = DOCS_ASSETS / "sphere1.nmesh.h5"
    assert downloadable_mesh.read_bytes() == FIXTURE_MESH.read_bytes()

    local_mesh = tmp_path / "sphere1.nmesh.h5"
    shutil.copyfile(downloadable_mesh, local_mesh)
    completed = _run_example(DOCS_EXAMPLES / "sphere_demag.py", tmp_path, local_mesh)

    assert "H_demag at the origin:" in completed.stdout
    with h5py.File(tmp_path / "results" / "sphere1_dat.h5") as output:
        field_at_origin = output["fields/H_demag"][:]
        assert field_at_origin.shape[1] == 3

    match = re.search(r"H_demag at the origin: (\[[^\n]+\]) A/m", completed.stdout)
    assert match is not None
    probe = ast.literal_eval(match.group(1))
    expected_x = -1.0e6 / 3.0
    assert abs(probe[0] / expected_x - 1.0) < 0.01
