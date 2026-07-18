use ndarray::{ArrayView1, ArrayView2};
use pyo3::prelude::*;
use std::f64::consts::PI;

use crate::common::{checked_indices, point3, Vec3};
use crate::lindholm::{
    boundary_node_solid_angles, lindholm_face_geometries,
    lindholm_triangle_contributions_precomputed, FaceGeometry,
};

pub(crate) struct ExactLindholmEvaluator {
    observers: Vec<Vec3>,
    faces: Vec<FaceGeometry>,
    incident_faces: Vec<Vec<(usize, usize)>>,
    diagonal: Vec<f64>,
}

impl ExactLindholmEvaluator {
    pub(crate) fn new(
        points: ArrayView2<'_, f64>,
        simplices: ArrayView2<'_, i64>,
        face_nodes: ArrayView2<'_, i64>,
        boundary_nodes: ArrayView1<'_, i64>,
        local_index_by_point: ArrayView1<'_, i64>,
    ) -> PyResult<Self> {
        let boundary_count = boundary_nodes.len();
        let observer_indices = checked_indices(boundary_nodes, points.shape()[0], "boundary node")?;
        let observers = observer_indices
            .iter()
            .map(|index| point3(points, *index))
            .collect::<Vec<_>>();
        let faces =
            lindholm_face_geometries(points, face_nodes, local_index_by_point, boundary_count)?;
        let mut incident_faces = vec![Vec::new(); boundary_count];
        for (face_index, face) in faces.iter().enumerate() {
            for local_index in 0..3 {
                incident_faces[face.local_indices[local_index]].push((face_index, local_index));
            }
        }
        let solid_angles = boundary_node_solid_angles(points, simplices)?;
        let diagonal = observer_indices
            .iter()
            .map(|index| solid_angles[*index] / (4.0 * PI) - 1.0)
            .collect();
        Ok(Self {
            observers,
            faces,
            incident_faces,
            diagonal,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.observers.len()
    }

    pub(crate) fn observer_points(&self) -> &[Vec3] {
        &self.observers
    }

    pub(crate) fn entry(&self, row: usize, column: usize) -> f64 {
        let observer = self.observers[row];
        let mut value = 0.0;
        for &(face_index, local_index) in &self.incident_faces[column] {
            let contributions =
                lindholm_triangle_contributions_precomputed(observer, &self.faces[face_index]);
            value += contributions[local_index];
        }
        if row == column {
            value += self.diagonal[row];
        }
        value
    }

    pub(crate) fn row_dot(&self, row: usize, values: &[f64]) -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(column, value)| self.entry(row, column) * value)
            .sum()
    }
}
