use crate::lindholm_hmatrix::entry::ExactLindholmEvaluator;

#[allow(clippy::too_many_arguments)]
pub(super) fn sampled_residual_pivot(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    u: &[f64],
    v: &[f64],
    rank: usize,
    used_rows: &[bool],
    used_columns: &[bool],
) -> Option<(usize, usize, f64)> {
    let sampled_rows = sample_positions(rows.len(), 12);
    let sampled_columns = sample_positions(columns.len(), 12);
    let mut best: Option<(usize, usize, f64)> = None;
    for row in sampled_rows {
        if used_rows[row] {
            continue;
        }
        for column in &sampled_columns {
            if used_columns[*column] {
                continue;
            }
            let approximation = (0..rank)
                .map(|factor| u[factor * rows.len() + row] * v[factor * columns.len() + *column])
                .sum::<f64>();
            let residual = evaluator.entry(rows[row], columns[*column]) - approximation;
            if best.is_none_or(|(_, _, value)| residual.abs() > value.abs()) {
                best = Some((row, *column, residual));
            }
        }
    }
    best
}

pub(super) fn residual_row(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    row: usize,
    u: &[f64],
    v: &[f64],
    rank: usize,
) -> Vec<f64> {
    let mut result = columns
        .iter()
        .map(|column| evaluator.entry(rows[row], *column))
        .collect::<Vec<_>>();
    for factor in 0..rank {
        let coefficient = u[factor * rows.len() + row];
        for column in 0..columns.len() {
            result[column] -= coefficient * v[factor * columns.len() + column];
        }
    }
    result
}

pub(super) fn residual_column(
    evaluator: &ExactLindholmEvaluator,
    rows: &[usize],
    columns: &[usize],
    column: usize,
    u: &[f64],
    v: &[f64],
    rank: usize,
) -> Vec<f64> {
    let mut result = rows
        .iter()
        .map(|row| evaluator.entry(*row, columns[column]))
        .collect::<Vec<_>>();
    for factor in 0..rank {
        let coefficient = v[factor * columns.len() + column];
        for row in 0..rows.len() {
            result[row] -= u[factor * rows.len() + row] * coefficient;
        }
    }
    result
}

pub(super) fn choose_reference(used: &[bool], rank: usize) -> Option<usize> {
    if used.is_empty() {
        return None;
    }
    let start = (used.len() / 2 + rank.saturating_mul(2654435761)) % used.len();
    (0..used.len())
        .map(|offset| (start + offset) % used.len())
        .find(|index| !used[*index])
}

pub(super) fn min_abs_unused(values: &[f64], used: &[bool]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(index, _)| !used[*index])
        .min_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .map(|(index, _)| index)
}

pub(super) fn max_abs_unused(values: &[f64], used: &[bool]) -> Option<(usize, f64)> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| !used[*index])
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
}

fn sample_positions(length: usize, count: usize) -> Vec<usize> {
    if length <= count {
        return (0..length).collect();
    }
    (0..count)
        .map(|index| index * (length - 1) / (count - 1))
        .collect()
}
