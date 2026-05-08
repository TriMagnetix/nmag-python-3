import pytest

from nmesh.backend import RawMesh
from nmesh.mesher.parity import (
    assert_canonical_mesh_equal,
    canonical_mesh_signature,
    compare_mesh_metrics,
    mesh_metric_summary,
    read_ascii_nmesh,
)


def _reference_triangle() -> RawMesh:
    return RawMesh(
        points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        simplices=[[0, 1, 2]],
        regions=[7],
        point_regions=[[7], [7], [7]],
        surfaces=[[0, 1], [0, 2], [1, 2]],
        links=[(0, 1), (0, 2), (1, 2)],
        periodic_point_indices=[[0, 1]],
        region_volumes=[0.5],
        dim=2,
    )


def test_canonical_mesh_signature_compares_topology_after_point_reordering():
    actual = RawMesh(
        points=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        simplices=[[2, 0, 1]],
        regions=[7],
        point_regions=[[7], [7], [7]],
        surfaces=[[2, 0], [1, 2], [0, 1]],
        links=[(2, 0), (1, 2), (0, 1)],
        periodic_point_indices=[[2, 0]],
        region_volumes=[0.5],
        dim=2,
    )

    assert_canonical_mesh_equal(actual, _reference_triangle())


def test_canonical_mesh_signature_is_exact_by_default():
    actual = _reference_triangle()
    expected = _reference_triangle()
    expected.points[1][0] = 1.0 + 1.0e-12

    with pytest.raises(AssertionError):
        assert_canonical_mesh_equal(actual, expected)


def test_canonical_mesh_signature_requires_explicit_coordinate_tolerance():
    actual = _reference_triangle()
    expected = _reference_triangle()
    expected.points[1][0] = 1.0 + 1.0e-12

    assert_canonical_mesh_equal(
        actual,
        expected,
        coordinate_tolerance=1.0e-10,
    )


def test_canonical_mesh_signature_rejects_ambiguous_tolerance():
    mesh = _reference_triangle()
    mesh.points[2] = [0.0, 0.5e-9]

    with pytest.raises(ValueError):
        canonical_mesh_signature(mesh, coordinate_tolerance=1.0e-9)


def test_read_ascii_nmesh_accepts_legacy_surface_and_periodic_rows(tmp_path):
    mesh_path = tmp_path / "legacy.nmesh"
    mesh_path.write_text(
        "\n".join(
            [
                "# PYFEM mesh file version 1.0",
                "# dim = 2 nodes = 3 simplices = 1 surfaces = 3 periodic = 1",
                "3",
                "0.0 0.0",
                "1.0 0.0",
                "0.0 1.0",
                "1",
                "7 0 1 2",
                "3",
                "7 -1 0 1",
                "7 -1 0 2",
                "7 -1 1 2",
                "1",
                "0 0 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    raw_mesh = read_ascii_nmesh(mesh_path)

    assert raw_mesh.points == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert raw_mesh.simplices == [[0, 1, 2]]
    assert raw_mesh.regions == [7]
    assert raw_mesh.surfaces == [[0, 1], [0, 2], [1, 2]]
    assert raw_mesh.periodic_point_indices == [[0, 1]]
    assert raw_mesh.links == [(0, 1), (0, 2), (1, 2)]
    assert raw_mesh.region_volumes == [0.5]


def test_mesh_metric_summary_reports_geometry_quality_metrics():
    summary = mesh_metric_summary(_reference_triangle())

    assert summary.dim == 2
    assert summary.point_count == 3
    assert summary.simplex_count == 1
    assert summary.region_counts == ((7, 1),)
    assert summary.region_volumes == ((7, 0.5),)
    assert summary.bbox_min == (0.0, 0.0)
    assert summary.bbox_max == (1.0, 1.0)
    assert summary.simplex_measure_min == 0.5
    assert summary.simplex_measure_mean == 0.5
    assert summary.simplex_measure_max == 0.5


def test_compare_mesh_metrics_allows_configured_metric_tolerances():
    actual = _reference_triangle()
    expected = _reference_triangle()
    expected.points[1][0] = 1.02
    expected.region_volumes = [0.51]

    comparison = compare_mesh_metrics(
        actual,
        expected,
        volume_relative_tolerance=0.05,
        length_relative_tolerance=0.05,
    )

    assert comparison.passed


def test_compare_mesh_metrics_reports_failures_outside_tolerance():
    actual = _reference_triangle()
    expected = _reference_triangle()
    expected.points[1][0] = 2.0
    expected.region_volumes = [1.0]

    comparison = compare_mesh_metrics(
        actual,
        expected,
        volume_relative_tolerance=0.05,
        length_relative_tolerance=0.05,
    )

    assert not comparison.passed
    assert any("bbox_max" in failure for failure in comparison.failures)
    assert any("region_volumes" in failure for failure in comparison.failures)
