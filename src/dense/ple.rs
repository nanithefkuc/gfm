//! `Ple` — the rank-revealing PLE/PLUQ decomposition, and the crate's single
//! elimination.
//!
//! `P·A·Q = L·U`, so `A = P⁻¹·L·U·Q⁻¹`, with `P` applying the recorded row
//! exchanges, `Q` the recorded column exchanges, `L`
//! unit lower `m × r`, and `U` upper `r × n`, stored in place: `lu` holds
//! `L`'s factors below the diagonal and `U` on and above it. Every query the
//! crate answers — rank, determinant, RREF, kernel, solve, inverse — is a
//! reader of this decomposition; there is exactly one pivoting loop per
//! storage domain.
//!
//! Pivot choice is free over a finite field (no numerical stability), so it
//! is chosen for locality: the first nonzero in the leftmost available
//! column, which keeps the permutations short and makes the decomposition
//! deterministic. The panel factorization shrinks the panel locally when a
//! slab is rank-deficient, so the pivot sequence — and therefore `lu`, `p`,
//! `q`, and both rank profiles — is identical to the unblocked sweep's,
//! byte for byte.
//!
//! [`Perm`] keeps row exchange an index operation; column exchange moves data
//! (one element swap per row), because kernels need row-contiguous columns.

use alloc::vec::Vec;
use core::fmt;

use fgf::{FieldBuffer, FieldElem};

use crate::backend::panel_width;
use crate::dense::{Matrix, Perm, ViewMut};

/// A rank-revealing `P·A·Q = L·U` decomposition.
///
/// The decomposed matrix had shape `m × n`; `rank() <= min(m, n)`. Rank
/// deficiency is normal: a rank-deficient input produces a rank-deficient
/// decomposition, not an error.
#[derive(Clone)]
pub struct Ple<F: FieldBuffer> {
    lu: Matrix<F>,
    p: Perm,
    q: Perm,
    rank: usize,
    row_profile: Vec<usize>,
    col_profile: Vec<usize>,
}

impl<F: FieldBuffer> fmt::Debug for Ple<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ple")
            .field("field", &F::NAME)
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

impl<F: FieldBuffer> Ple<F> {
    /// Decomposes `a`, consuming its storage.
    #[must_use]
    pub fn decompose(a: Matrix<F>) -> Self {
        let (rows, cols) = (a.rows(), a.cols());
        Self::decompose_impl(a, panel_width::<F>(rows, cols))
    }

    /// Reuses this decomposition's storage for another matrix of the same
    /// shape.
    ///
    /// `fill` receives a zeroed view of the decomposition's storage and must
    /// write the next input. The decomposition, permutation vectors, and
    /// rank-profile allocations are then recomputed in place. This is the
    /// allocation-free counterpart of repeated [`Ple::decompose`] calls for
    /// fixed-geometry workloads.
    pub fn redecompose(&mut self, fill: impl FnOnce(&mut ViewMut<'_, F>)) {
        let (rows, cols) = (self.rows(), self.cols());
        {
            let mut view = self.lu.as_view_mut();
            for row in 0..rows {
                view.row_mut(row).fill(0);
            }
            fill(&mut view);
        }
        self.redecompose_impl(panel_width::<F>(rows, cols));
    }

    /// Decomposes with an explicit panel width. The result is independent of
    /// the width, byte for byte.
    #[must_use]
    #[allow(dead_code)] // Reached through the `internals` facade.
    pub(crate) fn decompose_with_panel_width(a: Matrix<F>, panel_width: usize) -> Self {
        Self::decompose_impl(a, panel_width.max(1))
    }

    fn decompose_impl(mat: Matrix<F>, panel: usize) -> Self {
        let (rows, cols) = (mat.rows(), mat.cols());
        let mut decomposition = Self {
            lu: mat,
            p: Perm::identity(rows),
            q: Perm::identity(cols),
            rank: 0,
            row_profile: Vec::with_capacity(rows),
            col_profile: Vec::with_capacity(cols),
        };
        decomposition.redecompose_impl(panel);
        decomposition
    }

    fn redecompose_impl(&mut self, panel: usize) {
        let (rows, cols) = (self.lu.rows(), self.lu.cols());
        self.p.reset_identity();
        self.q.reset_identity();
        let limit = rows.min(cols);
        let mut frontier = 0;
        let mut wstart = 0;
        while frontier < limit && wstart < cols {
            let wend = (wstart + panel).min(cols);
            let pivots = factor_panel(
                &mut self.lu,
                &mut self.p,
                &mut self.q,
                frontier,
                wstart,
                wend,
            );
            if pivots == 0 {
                wstart = wend;
            } else {
                bulk_update(&mut self.lu, frontier, pivots, wend);
                frontier += pivots;
                wstart = frontier;
            }
        }
        self.rank = frontier;

        self.row_profile.clear();
        self.row_profile.extend(0..rows);
        self.p.apply(&mut self.row_profile);
        self.row_profile.truncate(self.rank);
        self.row_profile.sort_unstable();

        self.col_profile.clear();
        self.col_profile.extend(0..cols);
        self.q.apply(&mut self.col_profile);
        self.col_profile.truncate(self.rank);
        self.col_profile.sort_unstable();
    }

    /// The rank found by the decomposition.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Rows of the decomposed matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.lu.rows()
    }

    /// Columns of the decomposed matrix.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.lu.cols()
    }

    /// The rows holding the factorization's pivots, ascending: `rank()`
    /// independent row indices of the decomposed matrix. This is the pivot
    /// set the elimination chose, not necessarily the lexicographically
    /// smallest independent row set.
    #[must_use]
    pub fn pivot_rows(&self) -> &[usize] {
        &self.row_profile
    }

    /// The column rank profile: the lexicographically smallest set of
    /// independent column indices, ascending.
    #[must_use]
    pub fn col_rank_profile(&self) -> &[usize] {
        &self.col_profile
    }

    /// Reclaims the decomposed matrix's storage.
    #[must_use]
    pub fn into_matrix(self) -> Matrix<F> {
        self.lu
    }

    pub(crate) fn lu_matrix(&self) -> &Matrix<F> {
        &self.lu
    }

    pub(crate) fn p_perm(&self) -> &Perm {
        &self.p
    }

    pub(crate) fn q_perm(&self) -> &Perm {
        &self.q
    }
}

/// Factors one panel: up to `wend - wstart` pivot steps starting at row and
/// column position `k0`, with the pivot search restricted to column positions
/// `[step, wend)`. Returns the number of pivots found; fewer than requested
/// means every column in the window is zero below the frontier.
///
/// The search sees exactly the state the unblocked sweep would see: columns
/// in the window are updated by every elimination of this panel, and a column
/// once found zero below the frontier stays zero forever (the pivot rows
/// chosen later have zeros there by the scan's own invariant).
fn factor_panel<F: FieldBuffer>(
    mat: &mut Matrix<F>,
    row_perm: &mut Perm,
    col_perm: &mut Perm,
    frontier: usize,
    wstart: usize,
    wend: usize,
) -> usize {
    let rows = mat.rows();
    let limit = rows.min(mat.cols());
    let mut piv = frontier;
    // At most one window-width of pivots per panel, so the panel blocks stay
    // bounded; the window may also be exhausted first.
    while piv < limit && piv < wend && piv - frontier < wend - wstart {
        let Some((found_row, found_col)) = find_pivot(mat, piv, wstart, wend) else {
            break;
        };
        if found_col != piv {
            mat.swap_cols(found_col, piv);
            col_perm.record_swap(piv, found_col);
        }
        if found_row != piv {
            mat.swap_rows(found_row, piv);
            row_perm.record_swap(piv, found_row);
        }
        let pivot_inv = mat.get(piv, piv).inv();
        for row in (piv + 1)..rows {
            let entry = mat.get(row, piv);
            if entry.is_zero() {
                continue;
            }
            let factor = entry.mul(pivot_inv);
            mat.set(row, piv, factor);
            let (row_dst, row_src) = mat.two_live_rows(row, piv);
            let bytes = F::BYTES;
            crate::row_ops::mul_add::<F>(
                &mut row_dst[(piv + 1) * bytes..wend * bytes],
                factor.neg(),
                &row_src[(piv + 1) * bytes..wend * bytes],
            );
        }
        piv += 1;
    }
    piv - frontier
}

/// The first pivot in the unblocked sweep order: the leftmost column position
/// in `[max(step, wstart), wend)` with a nonzero entry at or below row
/// `step`, and that column's first nonzero row.
fn find_pivot<F: FieldBuffer>(
    mat: &Matrix<F>,
    step: usize,
    wstart: usize,
    wend: usize,
) -> Option<(usize, usize)> {
    for col in step.max(wstart)..wend {
        for row in step..mat.rows() {
            if !mat.get(row, col).is_zero() {
                return Some((row, col));
            }
        }
    }
    None
}

/// The trailing update after a panel: the triangular solve of the pivot
/// rows' right parts, then the rank-`s` update of the trailing submatrix.
/// Both cover only columns at and past `wend` — the window's columns were
/// already updated during the panel phase.
fn bulk_update<F: FieldBuffer>(mat: &mut Matrix<F>, frontier: usize, pivots: usize, wend: usize) {
    let cols = mat.cols();
    if wend >= cols {
        return;
    }
    // U12 = L11^-1 · A12, in place on the pivot rows.
    for row in frontier..(frontier + pivots) {
        for src in frontier..row {
            let factor = mat.get(row, src);
            if factor.is_zero() {
                continue;
            }
            let (row_dst, row_src) = mat.two_live_rows(row, src);
            let bytes = F::BYTES;
            crate::row_ops::mul_add::<F>(
                &mut row_dst[wend * bytes..cols * bytes],
                factor.neg(),
                &row_src[wend * bytes..cols * bytes],
            );
        }
    }
    update_trailing_axpy(mat, frontier, pivots, wend);
}

fn update_trailing_axpy<F: FieldBuffer>(
    mat: &mut Matrix<F>,
    frontier: usize,
    pivots: usize,
    wend: usize,
) {
    let (rows, cols) = (mat.rows(), mat.cols());
    for row in (frontier + pivots)..rows {
        for src in frontier..(frontier + pivots) {
            let factor = mat.get(row, src);
            if factor.is_zero() {
                continue;
            }
            let (row_dst, row_src) = mat.two_live_rows(row, src);
            let bytes = F::BYTES;
            crate::row_ops::mul_add::<F>(
                &mut row_dst[wend * bytes..cols * bytes],
                factor.neg(),
                &row_src[wend * bytes..cols * bytes],
            );
        }
    }
}
