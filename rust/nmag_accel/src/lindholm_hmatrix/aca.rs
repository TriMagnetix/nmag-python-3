mod pivot;
mod recompression;

use super::entry::ExactLindholmEvaluator;
use pivot::{
    choose_reference, max_abs_unused, min_abs_unused, residual_column, residual_row,
    sampled_residual_pivot,
};
pub(crate) use recompression::compress_block_exact_svd;
use recompression::recompress;

pub(crate) struct LowRankFactors {
    pub(crate) rank: usize,
    pub(crate) u: Vec<f64>,
    pub(crate) v: Vec<f64>,
}

pub(crate) fn compress_block(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    tolerance: f64,
    max_rank: usize,
) -> Option<LowRankFactors> {
    let limit = max_rank.min(rows.len()).min(columns.len());
    if limit == 0 {
        return Some(LowRankFactors {
            rank: 0,
            u: Vec::new(),
            v: Vec::new(),
        });
    }
    let mut u = Vec::with_capacity(rows.len() * limit);
    let mut v = Vec::with_capacity(columns.len() * limit);
    let mut used_rows = vec![false; rows.len()];
    let mut used_columns = vec![false; columns.len()];
    let mut approximation_norm = 0.0_f64;

    for rank in 0..limit {
        // ACA+ reference pivots are augmented with a distributed residual sample
        // so disconnected DLP normal directions cannot hide from the search.
        let reference_column = choose_reference(&used_columns, rank)?;
        let reference_column_values =
            residual_column(evaluator, rows, columns, reference_column, &u, &v, rank);
        let reference_row = min_abs_unused(&reference_column_values, &used_rows)?;
        let reference_row_values =
            residual_row(evaluator, rows, columns, reference_row, &u, &v, rank);
        let (column_candidate_row, column_candidate_value) =
            max_abs_unused(&reference_column_values, &used_rows)?;
        let (row_candidate_column, row_candidate_value) =
            max_abs_unused(&reference_row_values, &used_columns)?;
        let sampled_pivot = sampled_residual_pivot(
            evaluator,
            rows,
            columns,
            &u,
            &v,
            rank,
            &used_rows,
            &used_columns,
        );

        let (pivot_row, pivot_column, residual_row_values, residual_column_values) =
            if sampled_pivot.is_some_and(|(_, _, value)| {
                value.abs() > column_candidate_value.abs().max(row_candidate_value.abs())
            }) {
                let (sampled_row, _, _) = sampled_pivot?;
                let initial_row = residual_row(evaluator, rows, columns, sampled_row, &u, &v, rank);
                let (column, _) = max_abs_unused(&initial_row, &used_columns)?;
                let column_values = residual_column(evaluator, rows, columns, column, &u, &v, rank);
                let (row, _) = max_abs_unused(&column_values, &used_rows)?;
                let row_values = residual_row(evaluator, rows, columns, row, &u, &v, rank);
                (row, column, row_values, column_values)
            } else if column_candidate_value.abs() > row_candidate_value.abs() {
                let row_values =
                    residual_row(evaluator, rows, columns, column_candidate_row, &u, &v, rank);
                let (column, _) = max_abs_unused(&row_values, &used_columns)?;
                let column_values = residual_column(evaluator, rows, columns, column, &u, &v, rank);
                (column_candidate_row, column, row_values, column_values)
            } else {
                let column_values =
                    residual_column(evaluator, rows, columns, row_candidate_column, &u, &v, rank);
                let (row, _) = max_abs_unused(&column_values, &used_rows)?;
                let row_values = residual_row(evaluator, rows, columns, row, &u, &v, rank);
                (row, row_candidate_column, row_values, column_values)
            };

        let pivot = residual_column_values[pivot_row];
        if pivot.abs() <= f64::EPSILON {
            break;
        }
        let scaled_row = residual_row_values
            .iter()
            .map(|value| value / pivot)
            .collect::<Vec<_>>();
        u.extend_from_slice(&residual_column_values);
        v.extend_from_slice(&scaled_row);
        used_rows[pivot_row] = true;
        used_columns[pivot_column] = true;

        let term_norm = l2_norm(&residual_column_values) * l2_norm(&scaled_row);
        approximation_norm = approximation_norm.hypot(term_norm);
        if rank > 0 && term_norm <= tolerance * approximation_norm {
            let factors = recompress(
                LowRankFactors {
                    rank: rank + 1,
                    u,
                    v,
                },
                rows.len(),
                columns.len(),
                tolerance,
            )?;
            return validate_block(evaluator, rows, columns, &factors, tolerance)
                .then_some(factors);
        }
    }

    let factors = recompress(
        LowRankFactors {
            rank: u.len() / rows.len(),
            u,
            v,
        },
        rows.len(),
        columns.len(),
        tolerance,
    )?;
    validate_block(evaluator, rows, columns, &factors, tolerance).then_some(factors)
}

fn validate_block(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    factors: &LowRankFactors,
    tolerance: f64,
) -> bool {
    let row_samples = sample_positions(rows.len(), 7);
    let column_samples = sample_positions(columns.len(), 7);
    let mut exact_norm = 0.0_f64;
    let mut error_norm = 0.0_f64;
    for row in row_samples {
        for column in &column_samples {
            let exact = evaluator.entry(rows[row], columns[*column]);
            let approximate = (0..factors.rank)
                .map(|rank| {
                    factors.u[rank * rows.len() + row] * factors.v[rank * columns.len() + *column]
                })
                .sum::<f64>();
            exact_norm = exact_norm.hypot(exact);
            error_norm = error_norm.hypot(exact - approximate);
        }
    }
    error_norm <= tolerance * exact_norm.max(1.0e-30)
}

fn sample_positions(length: usize, count: usize) -> Vec<usize> {
    if length <= count {
        return (0..length).collect();
    }
    (0..count)
        .map(|index| index * (length - 1) / (count - 1))
        .collect()
}

fn l2_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value))
}
