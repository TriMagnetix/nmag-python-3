use crate::lindholm_hmatrix::aca::{compress_block, compress_block_exact_svd};
use crate::lindholm_hmatrix::cluster::ClusterTree;
use crate::lindholm_hmatrix::entry::ExactLindholmEvaluator;

use super::block::{Block, BlockValues};

const MAX_EXACT_RECOMPRESSION_ENTRIES: usize = 256 * 256;

pub(super) struct Builder<'a> {
    pub(super) evaluator: &'a ExactLindholmEvaluator,
    pub(super) tree: ClusterTree,
    pub(super) tolerance: f64,
    pub(super) eta: f64,
    pub(super) max_rank: usize,
    pub(super) memory_budget: usize,
    pub(super) storage_bytes: usize,
    pub(super) blocks: Vec<Block>,
}

impl Builder<'_> {
    pub(super) fn build_block(
        &mut self,
        row_index: usize,
        column_index: usize,
    ) -> Result<(), String> {
        let row_node = self.tree.nodes[row_index].clone();
        let column_node = self.tree.nodes[column_index].clone();
        let rows = self.tree.permutation[row_node.start..row_node.end].to_vec();
        let columns = self.tree.permutation[column_node.start..column_node.end].to_vec();

        if self.tree.admissible(row_index, column_index, self.eta) {
            let mut factors = compress_block(
                self.evaluator,
                &rows,
                &columns,
                self.tolerance,
                self.max_rank,
            );
            let dense_values = rows.len().saturating_mul(columns.len());
            if factors
                .as_ref()
                .is_none_or(|candidate| candidate.u.len() + candidate.v.len() >= dense_values)
                && dense_values <= MAX_EXACT_RECOMPRESSION_ENTRIES
            {
                factors = compress_block_exact_svd(
                    self.evaluator,
                    &rows,
                    &columns,
                    self.tolerance,
                    self.max_rank,
                );
            }
            if let Some(factors) = factors {
                let factor_values = factors.u.len() + factors.v.len();
                if factor_values < dense_values {
                    return self.push_block(Block {
                        row_start: row_node.start,
                        row_count: rows.len(),
                        column_start: column_node.start,
                        column_count: columns.len(),
                        values: BlockValues::LowRank(factors),
                    });
                }
            }
        }

        match (row_node.children, column_node.children) {
            (Some(row_children), Some(column_children)) => {
                for row_child in row_children {
                    for column_child in column_children {
                        self.build_block(row_child, column_child)?;
                    }
                }
                Ok(())
            }
            (Some(row_children), None) => {
                for row_child in row_children {
                    self.build_block(row_child, column_index)?;
                }
                Ok(())
            }
            (None, Some(column_children)) => {
                for column_child in column_children {
                    self.build_block(row_index, column_child)?;
                }
                Ok(())
            }
            (None, None) => {
                self.push_dense_block(&rows, &columns, row_node.start, column_node.start)
            }
        }
    }

    fn push_dense_block(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        row_start: usize,
        column_start: usize,
    ) -> Result<(), String> {
        let values = rows
            .iter()
            .flat_map(|row| {
                columns
                    .iter()
                    .map(|column| self.evaluator.entry(*row, *column))
            })
            .collect::<Vec<_>>();
        self.push_block(Block {
            row_start,
            row_count: rows.len(),
            column_start,
            column_count: columns.len(),
            values: BlockValues::Dense(values),
        })
    }

    fn push_block(&mut self, block: Block) -> Result<(), String> {
        let required = block.storage_bytes();
        let next = self
            .storage_bytes
            .checked_add(required)
            .ok_or_else(|| "hierarchical BEM storage accounting overflowed".to_owned())?;
        if next > self.memory_budget {
            return Err(format!(
                "hierarchical BEM memory budget exceeded: {next} bytes required, {} bytes allowed",
                self.memory_budget
            ));
        }
        self.storage_bytes = next;
        self.blocks.push(block);
        Ok(())
    }
}
