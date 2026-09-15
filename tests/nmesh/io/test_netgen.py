from __future__ import annotations

import gzip

import pytest

from nmesh.io.netgen import read_netgen_neutral

NETGEN = """4
0 0 0
1 0 0
0 1 0
0 0 1
1
7 1 2 3 4
"""


def test_read_netgen_neutral_converts_indices_and_preserves_regions(tmp_path):
    path = tmp_path / "mesh.mesh"
    path.write_text(NETGEN, encoding="utf-8")

    mesh = read_netgen_neutral(path)

    assert mesh.points == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert mesh.simplices == [[0, 1, 2, 3]]
    assert mesh.regions == [7]


def test_read_netgen_neutral_accepts_gzip(tmp_path):
    path = tmp_path / "mesh.mesh.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(NETGEN)

    assert read_netgen_neutral(path).simplices == [[0, 1, 2, 3]]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("1\n0 0 0\n1\n1 1 1 1 2\n", "outside"),
        ("4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1\n0 1 2 3 4\n", "region"),
        ("4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1\n1 1 2 3\n", "four point indices"),
    ],
)
def test_read_netgen_neutral_rejects_invalid_elements(tmp_path, content, message):
    path = tmp_path / "invalid.mesh"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_netgen_neutral(path)
