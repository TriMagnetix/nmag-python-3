use faer::{linalg::solvers::Svd, Mat};

use crate::lindholm_hmatrix::entry::ExactLindholmEvaluator;

use super::{validate_block, LowRankFactors};

pub(crate) fn compress_block_exact_svd(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    tolerance: f64,
    max_rank: usize,
) -> Option<LowRankFactors> {
    let matrix = Mat::from_fn(rows.len(), columns.len(), |row, column| {
        evaluator.entry(rows[row], columns[column])
    });
    let svd = Svd::new_thin(matrix.as_ref()).ok()?;
    let singular_values = svd.S().column_vector();
    let rank = truncated_rank(singular_values.iter().copied(), tolerance);
    if rank > max_rank {
        return None;
    }
    let left = svd.U();
    let right = svd.V();
    let mut u = vec![0.0; rows.len() * rank];
    let mut v = vec![0.0; columns.len() * rank];
    for factor in 0..rank {
        let scale = singular_values[factor].sqrt();
        for row in 0..rows.len() {
            u[factor * rows.len() + row] = left[(row, factor)] * scale;
        }
        for column in 0..columns.len() {
            v[factor * columns.len() + column] = right[(column, factor)] * scale;
        }
    }
    let factors = LowRankFactors { rank, u, v };
    validate_block(evaluator, rows, columns, &factors, tolerance).then_some(factors)
}

pub(super) fn recompress(
    factors: LowRankFactors,
    row_count: usize,
    column_count: usize,
    tolerance: f64,
) -> Option<LowRankFactors> {
    if factors.rank <= 1 {
        return Some(factors);
    }
    let rank = factors.rank;
    let u_matrix = Mat::from_fn(row_count, rank, |row, factor| {
        factors.u[factor * row_count + row]
    });
    let v_transpose = Mat::from_fn(column_count, rank, |column, factor| {
        factors.v[factor * column_count + column]
    });
    let u_qr = u_matrix.qr();
    let v_qr = v_transpose.qr();
    let q_u = u_qr.compute_thin_Q();
    let q_v = v_qr.compute_thin_Q();
    let r_u = u_qr.thin_R();
    let r_v = v_qr.thin_R();
    let core = Mat::from_fn(rank, rank, |row, column| {
        (0..rank)
            .map(|inner| r_u[(row, inner)] * r_v[(column, inner)])
            .sum::<f64>()
    });
    let svd = Svd::new_thin(core.as_ref()).ok()?;
    let singular_values = svd.S().column_vector();
    let recompressed_rank = truncated_rank(singular_values.iter().copied(), tolerance);
    let core_u = svd.U();
    let core_v = svd.V();
    let mut u = vec![0.0; row_count * recompressed_rank];
    let mut v = vec![0.0; column_count * recompressed_rank];
    for factor in 0..recompressed_rank {
        let scale = singular_values[factor].sqrt();
        for row in 0..row_count {
            u[factor * row_count + row] = scale
                * (0..rank)
                    .map(|inner| q_u[(row, inner)] * core_u[(inner, factor)])
                    .sum::<f64>();
        }
        for column in 0..column_count {
            v[factor * column_count + column] = scale
                * (0..rank)
                    .map(|inner| q_v[(column, inner)] * core_v[(inner, factor)])
                    .sum::<f64>();
        }
    }
    Some(LowRankFactors {
        rank: recompressed_rank,
        u,
        v,
    })
}

fn truncated_rank(values: impl Iterator<Item = f64>, tolerance: f64) -> usize {
    let values = values.collect::<Vec<_>>();
    let total_squared = values.iter().map(|value| value * value).sum::<f64>();
    if total_squared == 0.0 {
        return 0;
    }
    let target = tolerance * tolerance * total_squared;
    let mut tail_squared = total_squared;
    for (index, value) in values.iter().enumerate() {
        tail_squared = (tail_squared - value * value).max(0.0);
        if tail_squared <= target {
            return index + 1;
        }
    }
    values.len()
}
