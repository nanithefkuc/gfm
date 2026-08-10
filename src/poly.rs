//! Generic polynomial-row reduction.
//!
//! `gfm` owns the Mulders–Storjohann pivot schedule, collision handling, and
//! termination bound without owning a consumer's polynomial representation.
//! Consumers implement [`WeakPopovRow`] for their row type; coefficient storage
//! and allocation errors stay in that type.

use alloc::vec::Vec;
use core::fmt;

use fgf::FieldKernels;
use fgf::field::Elem;

use crate::ReduceError;

/// A leading polynomial term under a caller-supplied column shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PopovLeadingTerm {
    /// Degree before applying the column shift.
    pub degree: usize,
    /// Polynomial column containing the term.
    pub column: usize,
    /// `degree + shifts[column]`.
    pub shifted_degree: usize,
}

/// One row accepted by [`weak_popov`].
///
/// Implementations expose polynomial degrees and coefficients and perform one
/// shifted row update. `Error` carries both representation failures and
/// [`ReduceError`], so an allocation-aware consumer does not lose its native
/// error contract.
pub trait WeakPopovRow<F: FieldKernels> {
    /// Consumer error type.
    type Error: From<ReduceError>;

    /// Number of polynomial columns in this row.
    fn column_count(&self) -> usize;

    /// Degree of one polynomial column, or `None` when it is zero.
    fn degree(&self, column: usize) -> Option<usize>;

    /// Coefficient of `X^degree` in one polynomial column.
    fn coefficient(&self, column: usize, degree: usize) -> F::Elem;

    /// Leading term under `shifts`.
    ///
    /// Implementations with native leading-term metadata may override this
    /// default scan.
    ///
    /// # Errors
    ///
    /// Returns [`ReduceError::DegreeOverflow`] if a shifted degree cannot be
    /// represented.
    fn leading_term(&self, shifts: &[usize]) -> Result<Option<PopovLeadingTerm>, Self::Error>
    where
        Self: Sized,
    {
        leading_term::<F, Self>(self, shifts)
    }

    /// Adds `scale * X^shift * pivot` to this row.
    ///
    /// # Errors
    ///
    /// Returns the row type's error when the shifted update cannot be
    /// represented or applied.
    fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        pivot: &Self,
        shift: usize,
    ) -> Result<(), Self::Error>;
}

/// Reusable per-column collision-tracking storage for
/// [`weak_popov_with_scratch`].
///
/// [`weak_popov`] needs one leading-row slot per polynomial column. Holding
/// that buffer in a caller-owned scratch lets a consumer reduce many bases of
/// the same width in sequence without reallocating it: a warmed scratch reduces
/// with no `gfm`-owned allocation. The buffer is grow-only and its contents
/// carry no state between calls.
pub struct WeakPopovScratch {
    leading_rows: Vec<Option<usize>>,
}

impl WeakPopovScratch {
    /// A fresh, empty scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            leading_rows: Vec::new(),
        }
    }
}

impl Default for WeakPopovScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for WeakPopovScratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeakPopovScratch")
            .field("columns", &self.leading_rows.len())
            .finish()
    }
}

/// Reduces `basis` to shifted weak Popov form with Mulders–Storjohann row
/// reductions.
///
/// `shifts[column]` is added to every degree in that polynomial column. Ties in
/// shifted degree choose the larger column, making the result deterministic.
/// The iteration ceiling is the initial sum of
/// `columns * shifted_degree + leading_column + 1`: every valid row reduction
/// strictly decreases that measure, so reaching the ceiling signals a broken
/// row implementation rather than a difficult input.
///
/// # Errors
///
/// Returns the row's native error, [`ReduceError::ShiftCount`] for a shape
/// mismatch, [`ReduceError::DegreeOverflow`] when a shifted degree cannot be
/// represented, or [`ReduceError::Diverged`] if a row update fails to decrease
/// the termination measure.
pub fn weak_popov<F, R>(basis: &mut [R], shifts: &[usize]) -> Result<(), R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    let mut scratch = WeakPopovScratch::new();
    weak_popov_with_scratch::<F, R>(basis, shifts, &mut scratch)
}

/// Reduces `basis` to shifted weak Popov form, reusing `scratch` for the
/// per-column collision-tracking buffer.
///
/// This is identical in behavior and result to [`weak_popov`]; the only
/// difference is that the one buffer the reducer owns is taken from `scratch`
/// instead of allocated per call. Reusing a single scratch across reductions of
/// the same column count therefore performs no `gfm`-owned allocation after the
/// first, which lets a consumer reduce a stream of equally shaped bases in a
/// zero-allocation steady state. Consumer row updates still allocate through the
/// [`WeakPopovRow`] implementation.
///
/// # Errors
///
/// Returns the row's native error, [`ReduceError::ShiftCount`] for a shape
/// mismatch, [`ReduceError::DegreeOverflow`] when a shifted degree cannot be
/// represented, or [`ReduceError::Diverged`] if a row update fails to decrease
/// the termination measure.
pub fn weak_popov_with_scratch<F, R>(
    basis: &mut [R],
    shifts: &[usize],
    scratch: &mut WeakPopovScratch,
) -> Result<(), R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    let columns = shifts.len();
    scratch.leading_rows.clear();
    scratch.leading_rows.resize(columns, None);
    let leading_rows = &mut scratch.leading_rows;
    let mut ceiling = 0usize;
    let mut iterations = 0usize;
    loop {
        let initial_scan = iterations == 0;
        leading_rows.fill(None);
        let mut collision = None;
        for (row, polynomial) in basis.iter().enumerate() {
            if initial_scan {
                let row_columns = polynomial.column_count();
                if row_columns > columns {
                    return Err(ReduceError::ShiftCount {
                        columns: row_columns,
                        shifts: columns,
                    }
                    .into());
                }
            }
            let Some(leading) = polynomial.leading_term(shifts)? else {
                continue;
            };
            if initial_scan {
                let measure = columns
                    .saturating_mul(leading.shifted_degree)
                    .saturating_add(leading.column)
                    .saturating_add(1);
                ceiling = ceiling.saturating_add(measure);
            }
            if collision.is_some() {
                continue;
            }
            if let Some(previous) = leading_rows[leading.column] {
                collision = Some((previous, row));
                if !initial_scan {
                    break;
                }
            } else {
                leading_rows[leading.column] = Some(row);
            }
        }
        let Some((left, right)) = collision else {
            return Ok(());
        };
        if iterations >= ceiling {
            return Err(ReduceError::Diverged {
                iterations,
                ceiling,
            }
            .into());
        }
        reduce_pair::<F, R>(basis, left, right, shifts)?;
        iterations += 1;
    }
}

fn leading_term<F, R>(row: &R, shifts: &[usize]) -> Result<Option<PopovLeadingTerm>, R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    let mut leading = None;
    for (column, &shift) in shifts.iter().enumerate() {
        let Some(degree) = row.degree(column) else {
            continue;
        };
        let shifted_degree = degree
            .checked_add(shift)
            .ok_or(ReduceError::DegreeOverflow { degree, shift })?;
        let candidate = PopovLeadingTerm {
            degree,
            column,
            shifted_degree,
        };
        if leading.is_none_or(|current: PopovLeadingTerm| {
            (candidate.shifted_degree, candidate.column) > (current.shifted_degree, current.column)
        }) {
            leading = Some(candidate);
        }
    }
    Ok(leading)
}

fn reduce_pair<F, R>(
    basis: &mut [R],
    left: usize,
    right: usize,
    shifts: &[usize],
) -> Result<(), R::Error>
where
    F: FieldKernels,
    R: WeakPopovRow<F>,
{
    let left_leading = basis[left]
        .leading_term(shifts)?
        .expect("a colliding row has a leading term");
    let right_leading = basis[right]
        .leading_term(shifts)?
        .expect("a colliding row has a leading term");
    let (target, pivot, target_leading, pivot_leading) =
        if left_leading.degree >= right_leading.degree {
            (left, right, left_leading, right_leading)
        } else {
            (right, left, right_leading, left_leading)
        };
    let target_coefficient =
        basis[target].coefficient(target_leading.column, target_leading.degree);
    let pivot_coefficient = basis[pivot].coefficient(pivot_leading.column, pivot_leading.degree);
    let scale = target_coefficient.mul(pivot_coefficient.inv());
    let shift = target_leading.degree - pivot_leading.degree;
    let (target_row, pivot_row) = if target < pivot {
        let (lower, upper) = basis.split_at_mut(pivot);
        (&mut lower[target], &upper[0])
    } else {
        let (lower, upper) = basis.split_at_mut(target);
        (&mut upper[0], &lower[pivot])
    };
    target_row.add_scaled_shifted_assign(scale, pivot_row, shift)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use fgf::Gf8;
    use fgf::field::{Elem, Field};

    use super::*;

    type E = <Gf8 as Field>::Elem;

    struct Row {
        columns: Vec<Vec<E>>,
        update: bool,
    }

    impl WeakPopovRow<Gf8> for Row {
        type Error = ReduceError;

        fn column_count(&self) -> usize {
            self.columns.len()
        }

        fn degree(&self, column: usize) -> Option<usize> {
            self.columns[column]
                .iter()
                .rposition(|coefficient| !coefficient.is_zero())
        }

        fn coefficient(&self, column: usize, degree: usize) -> E {
            self.columns[column].get(degree).copied().unwrap_or(E::ZERO)
        }

        fn add_scaled_shifted_assign(
            &mut self,
            scale: E,
            pivot: &Self,
            shift: usize,
        ) -> Result<(), Self::Error> {
            if !self.update {
                return Ok(());
            }
            for (target, source) in self.columns.iter_mut().zip(&pivot.columns) {
                let required = source.len() + shift;
                target.resize(required.max(target.len()), E::ZERO);
                for (degree, &coefficient) in source.iter().enumerate() {
                    let position = degree + shift;
                    target[position] = target[position].add(scale.mul(coefficient));
                }
                while target
                    .last()
                    .is_some_and(|coefficient| coefficient.is_zero())
                {
                    target.pop();
                }
            }
            Ok(())
        }
    }

    fn row(columns: &[&[u8]], update: bool) -> Row {
        Row {
            columns: columns
                .iter()
                .map(|column| column.iter().map(|&value| E::from_raw(value)).collect())
                .collect(),
            update,
        }
    }

    #[test]
    fn resolves_leading_position_collision() {
        let mut basis = [row(&[&[0, 1], &[1]], true), row(&[&[1], &[]], true)];
        weak_popov::<Gf8, _>(&mut basis, &[0, 0]).unwrap();
        let leading: Vec<_> = basis
            .iter()
            .map(|row| {
                leading_term::<Gf8, _>(row, &[0, 0])
                    .unwrap()
                    .unwrap()
                    .column
            })
            .collect();
        assert_eq!(leading, [1, 0]);
    }

    #[test]
    fn non_decreasing_row_hits_proved_ceiling() {
        let mut basis = [row(&[&[0, 1], &[1]], false), row(&[&[1], &[]], false)];
        assert!(matches!(
            weak_popov::<Gf8, _>(&mut basis, &[0, 0]),
            Err(ReduceError::Diverged { .. })
        ));
    }

    #[test]
    fn scratch_variant_matches_plain_and_reuses_across_bases() {
        let leading_columns = |basis: &[Row]| -> Vec<usize> {
            basis
                .iter()
                .filter_map(|row| leading_term::<Gf8, _>(row, &[0, 0]).unwrap())
                .map(|term| term.column)
                .collect()
        };

        let mut plain = [row(&[&[0, 1], &[1]], true), row(&[&[1], &[]], true)];
        weak_popov::<Gf8, _>(&mut plain, &[0, 0]).unwrap();

        let mut scratch = WeakPopovScratch::new();
        let mut scratched = [row(&[&[0, 1], &[1]], true), row(&[&[1], &[]], true)];
        weak_popov_with_scratch::<Gf8, _>(&mut scratched, &[0, 0], &mut scratch).unwrap();
        assert_eq!(leading_columns(&scratched), leading_columns(&plain));

        // The same scratch reduces a second basis of the same column count.
        let mut second = [row(&[&[1, 1], &[1]], true), row(&[&[0, 1], &[1]], true)];
        weak_popov_with_scratch::<Gf8, _>(&mut second, &[0, 0], &mut scratch).unwrap();
        let mut second_plain = [row(&[&[1, 1], &[1]], true), row(&[&[0, 1], &[1]], true)];
        weak_popov::<Gf8, _>(&mut second_plain, &[0, 0]).unwrap();
        assert_eq!(leading_columns(&second), leading_columns(&second_plain));
    }
}
