mod block;
mod builder;

use rayon::prelude::*;

use super::cluster::ClusterTree;
use super::entry::ExactLindholmEvaluator;
use block::{Block, BlockValues};
use builder::Builder;

#[derive(Clone, Debug)]
pub(crate) struct OperatorDiagnostics {
    pub(crate) storage_bytes: usize,
    pub(crate) dense_blocks: usize,
    pub(crate) low_rank_blocks: usize,
    pub(crate) maximum_rank: usize,
    pub(crate) mean_rank: f64,
    pub(crate) sampled_relative_error: f64,
}

pub(crate) struct HMatrixOperator {
    size: usize,
    permutation: Vec<usize>,
    blocks: Vec<Block>,
    diagnostics: OperatorDiagnostics,
}

pub(crate) struct BuildOptions {
    pub(crate) tolerance: f64,
    pub(crate) eta: f64,
    pub(crate) leaf_size: usize,
    pub(crate) max_rank: usize,
    pub(crate) validation_vectors: usize,
    pub(crate) validation_rows: usize,
    pub(crate) memory_budget: usize,
}

impl HMatrixOperator {
    pub(crate) fn build(
        evaluator: &ExactLindholmEvaluator,
        options: BuildOptions,
    ) -> Result<Self, String> {
        let tree = ClusterTree::build(evaluator.observer_points(), options.leaf_size);
        let root = tree.root;
        let mut builder = Builder {
            evaluator,
            tree,
            tolerance: options.tolerance,
            eta: options.eta,
            max_rank: options.max_rank,
            memory_budget: options.memory_budget,
            storage_bytes: 0,
            blocks: Vec::new(),
        };
        builder.build_block(root, root)?;

        let ranks = builder
            .blocks
            .iter()
            .filter_map(|block| match &block.values {
                BlockValues::LowRank(factors) => Some(factors.rank),
                BlockValues::Dense(_) => None,
            })
            .collect::<Vec<_>>();
        let diagnostics = OperatorDiagnostics {
            storage_bytes: builder.storage_bytes,
            dense_blocks: builder
                .blocks
                .iter()
                .filter(|block| matches!(block.values, BlockValues::Dense(_)))
                .count(),
            low_rank_blocks: ranks.len(),
            maximum_rank: ranks.iter().copied().max().unwrap_or(0),
            mean_rank: if ranks.is_empty() {
                0.0
            } else {
                ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
            },
            sampled_relative_error: 0.0,
        };
        let mut operator = Self {
            size: evaluator.len(),
            permutation: builder.tree.permutation,
            blocks: builder.blocks,
            diagnostics,
        };
        let error = operator.certify(
            evaluator,
            options.validation_vectors,
            options.validation_rows,
        );
        operator.diagnostics.sampled_relative_error = error;
        if !error.is_finite() || error > options.tolerance {
            return Err(format!(
                "hierarchical BEM certification error {error:.6e} exceeds tolerance {:.6e}",
                options.tolerance
            ));
        }
        Ok(operator)
    }

    pub(crate) fn len(&self) -> usize {
        self.size
    }

    pub(crate) fn diagnostics(&self) -> &OperatorDiagnostics {
        &self.diagnostics
    }

    pub(crate) fn matvec(&self, input: &[f64]) -> Vec<f64> {
        let permuted_input = self
            .permutation
            .iter()
            .map(|index| input[*index])
            .collect::<Vec<_>>();
        let prepared = self
            .blocks
            .par_iter()
            .map(|block| block.prepare(&permuted_input))
            .collect::<Vec<_>>();
        let mut permuted_output = vec![0.0; self.size];
        permuted_output
            .par_chunks_mut(256)
            .enumerate()
            .for_each(|(chunk, output)| {
                let output_start = chunk * 256;
                for (block, coefficients) in self.blocks.iter().zip(&prepared) {
                    block.accumulate_rows(
                        &permuted_input,
                        coefficients.as_deref(),
                        output_start,
                        output,
                    );
                }
            });
        let mut output = vec![0.0; self.size];
        for (permuted, original) in self.permutation.iter().copied().enumerate() {
            output[original] = permuted_output[permuted];
        }
        output
    }

    fn certify(
        &self,
        evaluator: &ExactLindholmEvaluator,
        validation_vectors: usize,
        validation_rows: usize,
    ) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let rows = sample_positions(self.size, validation_rows)
            .into_iter()
            .map(|position| self.permutation[position])
            .collect::<Vec<_>>();
        let mut exact_norm = 0.0_f64;
        let mut error_norm = 0.0_f64;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..validation_vectors {
            let input = (0..self.size)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    if state & 1 == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect::<Vec<_>>();
            let approximate = self.matvec(&input);
            for row in &rows {
                let exact = evaluator.row_dot(*row, &input);
                exact_norm = exact_norm.hypot(exact);
                error_norm = error_norm.hypot(exact - approximate[*row]);
            }
        }
        error_norm / exact_norm.max(1.0e-30)
    }
}

fn sample_positions(length: usize, count: usize) -> Vec<usize> {
    let count = count.min(length);
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![length / 2];
    }
    (0..count)
        .map(|index| index * (length - 1) / (count - 1))
        .collect()
}
