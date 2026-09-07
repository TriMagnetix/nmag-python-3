use crate::lindholm_hmatrix::aca::LowRankFactors;

const BLOCK_METADATA_BYTES: usize = 8 * std::mem::size_of::<usize>();

pub(super) enum BlockValues {
    Dense(Vec<f64>),
    LowRank(LowRankFactors),
}

pub(super) struct Block {
    pub(super) row_start: usize,
    pub(super) row_count: usize,
    pub(super) column_start: usize,
    pub(super) column_count: usize,
    pub(super) values: BlockValues,
}

impl Block {
    pub(super) fn storage_bytes(&self) -> usize {
        BLOCK_METADATA_BYTES
            + match &self.values {
                BlockValues::Dense(values) => values.len() * std::mem::size_of::<f64>(),
                BlockValues::LowRank(factors) => {
                    (factors.u.len() + factors.v.len()) * std::mem::size_of::<f64>()
                }
            }
    }

    pub(super) fn prepare(&self, input: &[f64]) -> Option<Vec<f64>> {
        let columns = &input[self.column_start..self.column_start + self.column_count];
        match &self.values {
            BlockValues::Dense(_) => None,
            BlockValues::LowRank(factors) => Some(
                (0..factors.rank)
                    .map(|rank| {
                        factors.v[rank * self.column_count..(rank + 1) * self.column_count]
                            .iter()
                            .zip(columns)
                            .map(|(entry, value)| entry * value)
                            .sum::<f64>()
                    })
                    .collect(),
            ),
        }
    }

    pub(super) fn accumulate_rows(
        &self,
        input: &[f64],
        coefficients: Option<&[f64]>,
        output_start: usize,
        output: &mut [f64],
    ) {
        let overlap_start = output_start.max(self.row_start);
        let overlap_end = (output_start + output.len()).min(self.row_start + self.row_count);
        if overlap_start >= overlap_end {
            return;
        }
        let columns = &input[self.column_start..self.column_start + self.column_count];
        for global_row in overlap_start..overlap_end {
            let block_row = global_row - self.row_start;
            let value: f64 = match &self.values {
                BlockValues::Dense(values) => values
                    [block_row * self.column_count..(block_row + 1) * self.column_count]
                    .iter()
                    .zip(columns)
                    .map(|(entry, value)| entry * value)
                    .sum(),
                BlockValues::LowRank(factors) => {
                    let coefficients = coefficients.expect("low-rank coefficients are prepared");
                    (0..factors.rank)
                        .map(|rank| {
                            factors.u[rank * self.row_count + block_row] * coefficients[rank]
                        })
                        .sum()
                }
            };
            output[global_row - output_start] += value;
        }
    }
}
