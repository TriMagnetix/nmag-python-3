use crate::common::Vec3;

#[derive(Clone, Debug)]
pub(crate) struct ClusterNode {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) lower: Vec3,
    pub(crate) upper: Vec3,
    pub(crate) children: Option<[usize; 2]>,
}

impl ClusterNode {
    pub(crate) fn diameter(&self) -> f64 {
        let dx = self.upper[0] - self.lower[0];
        let dy = self.upper[1] - self.lower[1];
        let dz = self.upper[2] - self.lower[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

pub(crate) struct ClusterTree {
    pub(crate) permutation: Vec<usize>,
    pub(crate) nodes: Vec<ClusterNode>,
    pub(crate) root: usize,
}

impl ClusterTree {
    pub(crate) fn build(points: &[Vec3], leaf_size: usize) -> Self {
        let mut tree = Self {
            permutation: (0..points.len()).collect(),
            nodes: Vec::new(),
            root: 0,
        };
        tree.root = tree.build_node(points, 0, points.len(), leaf_size);
        tree
    }

    fn build_node(&mut self, points: &[Vec3], start: usize, end: usize, leaf_size: usize) -> usize {
        let (lower, upper) = bounds(points, &self.permutation[start..end]);
        let node_index = self.nodes.len();
        self.nodes.push(ClusterNode {
            start,
            end,
            lower,
            upper,
            children: None,
        });
        if end - start <= leaf_size {
            return node_index;
        }

        let spans = [
            upper[0] - lower[0],
            upper[1] - lower[1],
            upper[2] - lower[2],
        ];
        let axis = if spans[1] > spans[0] && spans[1] >= spans[2] {
            1
        } else if spans[2] > spans[0] && spans[2] > spans[1] {
            2
        } else {
            0
        };
        self.permutation[start..end].sort_by(|left, right| {
            points[*left][axis]
                .total_cmp(&points[*right][axis])
                .then_with(|| left.cmp(right))
        });
        let middle = start + (end - start) / 2;
        let left = self.build_node(points, start, middle, leaf_size);
        let right = self.build_node(points, middle, end, leaf_size);
        self.nodes[node_index].children = Some([left, right]);
        node_index
    }

    pub(crate) fn admissible(&self, row: usize, column: usize, eta: f64) -> bool {
        let row_node = &self.nodes[row];
        let column_node = &self.nodes[column];
        let distance = box_distance(row_node, column_node);
        distance > 0.0 && row_node.diameter().max(column_node.diameter()) <= eta * distance
    }
}

fn bounds(points: &[Vec3], indices: &[usize]) -> (Vec3, Vec3) {
    if indices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut lower = points[indices[0]];
    let mut upper = lower;
    for index in &indices[1..] {
        for axis in 0..3 {
            lower[axis] = lower[axis].min(points[*index][axis]);
            upper[axis] = upper[axis].max(points[*index][axis]);
        }
    }
    (lower, upper)
}

fn box_distance(left: &ClusterNode, right: &ClusterNode) -> f64 {
    let mut squared = 0.0;
    for axis in 0..3 {
        let gap = if left.upper[axis] < right.lower[axis] {
            right.lower[axis] - left.upper[axis]
        } else if right.upper[axis] < left.lower[axis] {
            left.lower[axis] - right.upper[axis]
        } else {
            0.0
        };
        squared += gap * gap;
    }
    squared.sqrt()
}
