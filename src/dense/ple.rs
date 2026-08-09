//! `Ple` — the rank-revealing PLE/PLUQ decomposition, and the crate's single
//! elimination.
//!
//! `A = P·L·U·Q` with `P` a row permutation, `Q` a column permutation, `L`
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
use core::marker::PhantomData;

#[cfg(feature = "internals")]
use fgf::ops;
use fgf::{FieldKernels, field::Elem};

use crate::backend::panel_width;
use crate::dense::{Matrix, Perm};

/// A rank-revealing `A = P·L·U·Q` decomposition.
///
/// The decomposed matrix had shape `m × n`; `rank() <= min(m, n)`. Rank
/// deficiency is normal: a rank-deficient input produces a rank-deficient
/// decomposition, not an error.
#[derive(Clone)]
pub struct Ple<F: FieldKernels> {
    lu: Matrix<F>,
    p: Perm,
    q: Perm,
    rank: usize,
    row_profile: Vec<usize>,
    col_profile: Vec<usize>,
}

/// Reusable workspace for [`Ple::decompose`].
///
/// The Newton–John candidate table is grow-only. The normal blocked/AXPY path
/// leaves it untouched.
pub struct PleScratch<F: FieldKernels> {
    table: Vec<u8>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> PleScratch<F> {
    /// An empty scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            table: Vec::new(),
            field: PhantomData,
        }
    }
}

impl<F: FieldKernels> Default for PleScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldKernels> fmt::Debug for PleScratch<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PleScratch")
            .field("table_bytes", &self.table.len())
            .finish_non_exhaustive()
    }
}

impl<F: FieldKernels> fmt::Debug for Ple<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ple")
            .field("field", &F::NAME)
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
enum TrailingMode {
    Axpy,
    #[cfg(feature = "internals")]
    NewtonJohn,
}

impl<F: FieldKernels> Ple<F> {
    /// Decomposes `a`, consuming its storage.
    #[must_use]
    pub fn decompose(a: Matrix<F>, scratch: &mut PleScratch<F>) -> Self {
        let (rows, cols) = (a.rows(), a.cols());
        Self::decompose_impl(a, panel_width::<F>(rows, cols), TrailingMode::Axpy, scratch)
    }

    /// Decomposes with an explicit panel width. The result is independent of
    /// the width, byte for byte.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn decompose_with_panel_width(
        a: Matrix<F>,
        scratch: &mut PleScratch<F>,
        panel_width: usize,
    ) -> Self {
        Self::decompose_impl(a, panel_width.max(1), TrailingMode::Axpy, scratch)
    }
    /// Forces the Newton–John trailing-update candidate.
    ///
    /// # Panics
    ///
    /// Panics unless field elements occupy one byte.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn decompose_newton_john(a: Matrix<F>, scratch: &mut PleScratch<F>) -> Self {
        assert_eq!(F::BYTES, 1, "Newton-John table requires one-byte elements");
        let (rows, cols) = (a.rows(), a.cols());
        Self::decompose_impl(
            a,
            panel_width::<F>(rows, cols),
            TrailingMode::NewtonJohn,
            scratch,
        )
    }

    fn decompose_impl(
        mat: Matrix<F>,
        panel: usize,
        mode: TrailingMode,
        scratch: &mut PleScratch<F>,
    ) -> Self {
        let (rows, cols) = (mat.rows(), mat.cols());
        let mut decomposition = Self {
            lu: mat,
            p: Perm::identity(rows),
            q: Perm::identity(cols),
            rank: 0,
            row_profile: Vec::with_capacity(rows),
            col_profile: Vec::with_capacity(cols),
        };
        decomposition.redecompose_impl(panel, mode, scratch);
        decomposition
    }
    pub(crate) fn matrix_for_redecomposition(&mut self) -> &mut Matrix<F> {
        &mut self.lu
    }

    pub(crate) fn redecompose(&mut self, scratch: &mut PleScratch<F>) {
        let (rows, cols) = (self.rows(), self.cols());
        self.redecompose_impl(panel_width::<F>(rows, cols), TrailingMode::Axpy, scratch);
    }

    fn redecompose_impl(&mut self, panel: usize, mode: TrailingMode, scratch: &mut PleScratch<F>) {
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
                bulk_update(&mut self.lu, frontier, pivots, wend, mode, scratch);
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

    /// The row rank profile: the lexicographically smallest set of
    /// independent row indices, ascending.
    #[must_use]
    pub fn row_rank_profile(&self) -> &[usize] {
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

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<F: FieldKernels> Ple<F> {
    /// The in-place `L`/`U` storage: factors below the diagonal, `U` on and
    /// above it, in the eliminated row and column order.
    #[must_use]
    pub fn lu(&self) -> &Matrix<F> {
        self.lu_matrix()
    }

    /// The row permutation, as a LAPACK-style swap list.
    #[must_use]
    pub fn p(&self) -> &Perm {
        self.p_perm()
    }

    /// The column permutation, as a LAPACK-style swap list.
    #[must_use]
    pub fn q(&self) -> &Perm {
        self.q_perm()
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
fn factor_panel<F: FieldKernels>(
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
                factor,
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
fn find_pivot<F: FieldKernels>(
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
fn bulk_update<F: FieldKernels>(
    mat: &mut Matrix<F>,
    frontier: usize,
    pivots: usize,
    wend: usize,
    mode: TrailingMode,
    scratch: &mut PleScratch<F>,
) {
    #[cfg(not(feature = "internals"))]
    let _ = scratch;
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
                factor,
                &row_src[wend * bytes..cols * bytes],
            );
        }
    }
    match mode {
        TrailingMode::Axpy => update_trailing_axpy(mat, frontier, pivots, wend),
        #[cfg(feature = "internals")]
        TrailingMode::NewtonJohn => {
            update_trailing_newton_john(mat, frontier, pivots, wend, scratch);
        }
    }
}

fn update_trailing_axpy<F: FieldKernels>(
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
                factor,
                &row_src[wend * bytes..cols * bytes],
            );
        }
    }
}

#[cfg(feature = "internals")]
/// One 256-entry multiplication table per pivot row. Table construction uses
/// `fgf::ops`; applying a selected multiple is the field's XOR-add kernel.
fn update_trailing_newton_john<F: FieldKernels>(
    mat: &mut Matrix<F>,
    frontier: usize,
    pivots: usize,
    wend: usize,
    scratch: &mut PleScratch<F>,
) {
    debug_assert_eq!(F::BYTES, 1);
    let (rows, cols) = (mat.rows(), mat.cols());
    let tail = cols - wend;
    scratch.table.resize(256 * tail, 0);
    for src in frontier..(frontier + pivots) {
        scratch.table.fill(0);
        let pivot_tail = &mat.row(src)[wend..cols];
        for coefficient in 1..=u8::MAX {
            let start = usize::from(coefficient) * tail;
            let multiple = &mut scratch.table[start..start + tail];
            multiple.copy_from_slice(pivot_tail);
            ops::mul_assign::<F>(multiple, F::read(&[coefficient]));
        }
        for row in (frontier + pivots)..rows {
            let factor = mat.get(row, src);
            if factor.is_zero() {
                continue;
            }
            let mut key = [0u8; 1];
            F::write(&mut key, factor);
            let start = usize::from(key[0]) * tail;
            ops::add_assign::<F>(
                &mut mat.row_mut(row)[wend..cols],
                &scratch.table[start..start + tail],
            );
        }
    }
}
