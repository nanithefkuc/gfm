//! The derived queries: every one a reader of a cached [`Ple`], none an
//! elimination of its own (the one-elimination invariant).
//!
//! With `A = P·L·U·Q`, `P·A·Q = L·U`, and all arithmetic in characteristic
//! two:
//!
//! - `det` is the product of `U`'s diagonal — the permutation signs are
//!   always one in characteristic two, and the product is zero exactly when
//!   the rank is short of the order;
//! - a solve is `x = Q·U⁻¹·L⁻¹·P·b`: permute, forward-substitute the unit
//!   triangle, check the tail for consistency, back-substitute, permute;
//! - an inverse is the same pipeline applied to the identity;
//! - the kernel basis lives at the free column positions of the permuted
//!   system, back-substituted through `U`, then un-permuted.

use fgf::{FieldKernels, field::Elem, ops};

use crate::SolveError;
use crate::dense::{Matrix, Ple};

/// Reusable workspace for [`Ple::solve_into`].
///
/// Holds the eliminated right-hand side (`m × s`). Grow-only: solving
/// repeatedly with the same shape allocates nothing.
pub struct SolveScratch<F: FieldKernels> {
    ws: Option<Matrix<F>>,
}

impl<F: FieldKernels> SolveScratch<F> {
    /// An empty scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self { ws: None }
    }

    fn ensure(&mut self, rows: usize, cols: usize) -> &mut Matrix<F> {
        let needs_new = match &self.ws {
            Some(ws) => ws.rows() != rows || ws.cols() != cols,
            None => true,
        };
        if needs_new {
            self.ws = Some(
                Matrix::zeros(rows, cols)
                    .unwrap_or_else(|_| unreachable!("shape comes from an existing matrix")),
            );
        }
        self.ws.as_mut().unwrap_or_else(|| unreachable!())
    }
}

impl<F: FieldKernels> Default for SolveScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldKernels> core::fmt::Debug for SolveScratch<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SolveScratch").finish_non_exhaustive()
    }
}

impl<F: FieldKernels> Ple<F> {
    /// The determinant: the product of `U`'s diagonal for a full-rank square
    /// matrix, zero otherwise. Permutation signs are one in characteristic
    /// two, so neither permutation contributes a factor.
    ///
    /// # Panics
    ///
    /// Panics if the decomposed matrix is not square.
    #[must_use]
    pub fn det(&self) -> F::Elem {
        assert_eq!(self.rows(), self.cols(), "det requires a square matrix");
        if self.rank() < self.rows() {
            return F::Elem::ZERO;
        }
        let mut d = F::Elem::ONE;
        let lu = self.lu_matrix();
        for k in 0..self.rank() {
            d = d.mul(lu.get(k, k));
        }
        d
    }

    /// The reduced row echelon form of the decomposed matrix, written into
    /// `out` (`m × n`, bottom `m - rank` rows zero).
    ///
    /// `out` is filled from the echelon form with the column permutation
    /// undone — RREF is not invariant under column permutation, so the
    /// reduction runs in the original column order — and then Gauss–Jordan
    /// back-substitution through the pivot columns.
    ///
    /// # Panics
    ///
    /// Panics unless `out` is `m × n`.
    pub fn rref_into(&self, out: &mut Matrix<F>) {
        assert_eq!(
            (out.rows(), out.cols()),
            (self.rows(), self.cols()),
            "rref output shape mismatch"
        );
        let (rows, rank) = (self.rows(), self.rank());
        for row in 0..rows {
            out.row_mut(row).fill(0);
        }
        let lu = self.lu_matrix();
        for piv in 0..rank {
            // Copy the U echelon row; the stored L multipliers below the
            // diagonal (permuted columns `0..piv`) are not part of the
            // echelon form and must not enter the back-substitution.
            out.row_mut(piv).copy_from_slice(lu.row(piv));
            out.row_mut(piv)[..piv * F::BYTES].fill(0);
        }
        // Undo the column permutation so the reduction runs in the original
        // column order; RREF(E·Q⁻¹) is not RREF(E)·Q⁻¹.
        out.apply_col_perm_inv(self.q_perm());
        for piv in (0..rank).rev() {
            let pivot_col = self.q_perm().image_at(piv);
            let pivot_val = out.get(piv, pivot_col);
            if !pivot_val.is_one() {
                ops::mul_assign::<F>(out.row_mut(piv), pivot_val.inv());
            }
            for row in 0..piv {
                let factor = out.get(row, pivot_col);
                if factor.is_zero() {
                    continue;
                }
                let (row_dst, row_src) = out.two_live_rows(row, piv);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
        }
    }

    /// A basis of the right kernel, written into `out` (`n × (n - rank)`),
    /// one basis vector per column.
    ///
    /// Basis column `j` corresponds to the `(rank + j)`-th column of the
    /// permuted system: it has a one at the original index of that column
    /// and zeros at every other free position. This is the documented basis
    /// convention; the GF(2) domain produces the same one.
    ///
    /// # Panics
    ///
    /// Panics unless `out` is `n × (n - rank)`.
    pub fn kernel_into(&self, out: &mut Matrix<F>) {
        let (cols, rank) = (self.cols(), self.rank());
        assert_eq!(
            (out.rows(), out.cols()),
            (cols, cols - rank),
            "kernel output shape mismatch"
        );
        let lu = self.lu_matrix();
        for row in 0..cols {
            out.row_mut(row).fill(0);
        }
        for col in 0..(cols - rank) {
            let free = rank + col;
            // Work down the permuted positions, writing column `col`.
            for piv in (0..rank).rev() {
                let mut acc = lu.get(piv, free);
                for term in (piv + 1)..rank {
                    let already = out.get(term, col);
                    if already.is_zero() {
                        continue;
                    }
                    acc = acc.add(lu.get(piv, term).mul(already));
                }
                let pivot = lu.get(piv, piv);
                out.set(piv, col, acc.mul(pivot.inv()));
            }
            out.set(free, col, F::Elem::ONE);
        }
        // x = Q·z: un-permute the solution entries, one row permutation for
        // the whole basis matrix.
        out.apply_row_perm_inv(self.q_perm())
            .expect("row count matches by construction");
    }

    /// Solves `A·x = b` for `b` (`m × s`), writing `x` (`n × s`).
    ///
    /// Rank-deficient systems are answered with the solution whose free
    /// variables are zero.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system is inconsistent, naming the
    /// first row of the eliminated system that is zero on the left and
    /// nonzero on the right; `out` is left untouched in that case.
    ///
    /// # Panics
    ///
    /// Panics unless `rhs` is `m × s` and `out` is `n × s` for some `s`.
    pub fn solve_into(
        &self,
        rhs: &Matrix<F>,
        out: &mut Matrix<F>,
        scratch: &mut SolveScratch<F>,
    ) -> Result<(), SolveError> {
        let (rows, cols, rank) = (self.rows(), self.cols(), self.rank());
        assert_eq!(rhs.rows(), rows, "right-hand side row mismatch");
        assert_eq!(out.rows(), cols, "solution row mismatch");
        assert_eq!(rhs.cols(), out.cols(), "solution column mismatch");
        let nrhs = rhs.cols();
        let lu = self.lu_matrix();
        let ws = scratch.ensure(rows, nrhs);
        for row in 0..rows {
            ws.row_mut(row).copy_from_slice(rhs.row(row));
        }
        ws.apply_row_perm(self.p_perm())
            .expect("row count matches by construction");
        // Forward substitution through the unit triangle L: b̂ = L⁻¹(P·b).
        for row in 0..rows {
            let top = row.min(rank);
            for src in 0..top {
                let factor = lu.get(row, src);
                if factor.is_zero() {
                    continue;
                }
                let (row_dst, row_src) = ws.two_live_rows(row, src);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
        }
        // Consistency: the eliminated tail rows must vanish on the right too.
        for row in rank..rows {
            if ws.row(row).iter().any(|&v| v != 0) {
                return Err(SolveError::Inconsistent { row });
            }
        }
        // Back substitution through U into the permuted solution order.
        for row in 0..cols {
            out.row_mut(row).fill(0);
        }
        for piv in (0..rank).rev() {
            out.row_mut(piv).copy_from_slice(ws.row(piv));
            for term in (piv + 1)..rank {
                let factor = lu.get(piv, term);
                if factor.is_zero() {
                    continue;
                }
                let (row_dst, row_src) = out.two_live_rows(piv, term);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
            let pivot = lu.get(piv, piv);
            if !pivot.is_one() {
                ops::mul_assign::<F>(out.row_mut(piv), pivot.inv());
            }
        }
        out.apply_row_perm_inv(self.q_perm())
            .expect("row count matches by construction");
        Ok(())
    }

    /// Inverts the decomposed matrix, writing `A⁻¹` into `out` (`n × n`).
    ///
    /// # Errors
    ///
    /// [`SolveError::Singular`] with the found rank and the order if the
    /// matrix is rank-deficient; `out` is left untouched in that case.
    ///
    /// # Panics
    ///
    /// Panics unless the decomposed matrix and `out` are square of order `n`.
    pub fn inverse_into(&self, out: &mut Matrix<F>) -> Result<(), SolveError> {
        assert_eq!(self.rows(), self.cols(), "inverse requires a square matrix");
        let order = self.rows();
        assert_eq!(
            (out.rows(), out.cols()),
            (order, order),
            "inverse output shape mismatch"
        );
        let rank = self.rank();
        if rank < order {
            return Err(SolveError::Singular { rank, order });
        }
        let lu = self.lu_matrix();
        // out := I, then X = Q·U⁻¹·L⁻¹·P applied in place.
        for row in 0..order {
            out.row_mut(row).fill(0);
            out.set(row, row, F::Elem::ONE);
        }
        out.apply_row_perm(self.p_perm())
            .expect("row count matches by construction");
        for row in 0..order {
            for src in 0..row {
                let factor = lu.get(row, src);
                if factor.is_zero() {
                    continue;
                }
                let (row_dst, row_src) = out.two_live_rows(row, src);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
        }
        for piv in (0..order).rev() {
            for term in (piv + 1)..order {
                let factor = lu.get(piv, term);
                if factor.is_zero() {
                    continue;
                }
                let (row_dst, row_src) = out.two_live_rows(piv, term);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
            let pivot = lu.get(piv, piv);
            if !pivot.is_one() {
                ops::mul_assign::<F>(out.row_mut(piv), pivot.inv());
            }
        }
        out.apply_row_perm_inv(self.q_perm())
            .expect("row count matches by construction");
        Ok(())
    }
}
