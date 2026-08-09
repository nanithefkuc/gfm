//! `Ple` over GF(2) — the bit domain's single elimination.
//!
//! Structurally the twin of [`crate::dense::Ple`]: the same `A = P·L·U·Q`
//! decomposition, the same pivot order (leftmost available column, first
//! nonzero row), the same panel-blocked schedule with a locally shrinking
//! window on rank-deficient slabs, and therefore the same `lu`, `p`, `q`,
//! and rank profiles. Only the inner loop differs (I8): a bit is one bit, a
//! row is `u64` words, a pivot is always one, no coefficient exists, and the
//! elimination of a column is a masked word XOR of the pivot row rather than
//! an AXPY. Normalization is a no-op — the only nonzero element is one.
//!
//! No Method-of-the-Four-Russians slab tables yet: that acceleration keeps
//! the same answer and lands later, gated on measurement.

use alloc::vec::Vec;
use core::fmt;

use crate::bits::BitMatrix;
use crate::dense::Perm;

/// Eight pivots form a 256-entry M4RI combination table. Three pinned runs on
/// this host put the crossover at 128 rows/columns: the table path loses at 64
/// and wins at 128 and above.
/// See `BENCHMARKS.md` for the three-run boundary measurements.
const SLAB_WIDTH: usize = 8;
const M4RI_CROSSOVER: usize = 128;

/// A rank-revealing `A = P·L·U·Q` decomposition over GF(2).
///
/// The decomposed matrix had shape `m × n`; `rank() <= min(m, n)`. Rank
/// deficiency is normal, not an error.
#[derive(Clone)]
pub struct Ple {
    lu: BitMatrix,
    p: Perm,
    q: Perm,
    rank: usize,
    row_profile: Vec<usize>,
    col_profile: Vec<usize>,
}

/// Reusable workspace for [`Ple::decompose`].
///
/// The table is grow-only. Reusing the scratch for repeated decompositions of
/// the same geometry avoids rebuilding its allocation.
pub struct PleScratch {
    table: Vec<u64>,
}

impl PleScratch {
    /// A fresh scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self { table: Vec::new() }
    }
}

impl Default for PleScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for PleScratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PleScratch")
            .field("table_words", &self.table.len())
            .finish()
    }
}

impl fmt::Debug for Ple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ple")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
enum BulkMode {
    Plain,
    M4ri,
}

impl Ple {
    /// Decomposes `a`, consuming its storage.
    #[must_use]
    pub fn decompose(a: BitMatrix, scratch: &mut PleScratch) -> Self {
        let mode = if a.rows().min(a.cols()) >= M4RI_CROSSOVER {
            BulkMode::M4ri
        } else {
            BulkMode::Plain
        };
        Self::decompose_impl(a, SLAB_WIDTH, mode, scratch)
    }

    /// Decomposes with an explicit panel width. The result is independent of
    /// the width, byte for byte.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn decompose_with_panel_width(
        a: BitMatrix,
        scratch: &mut PleScratch,
        panel_width: usize,
    ) -> Self {
        Self::decompose_impl(a, panel_width.max(1), BulkMode::Plain, scratch)
    }

    /// Forces the untabled AXPY-style trailing update.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn decompose_plain(a: BitMatrix, scratch: &mut PleScratch) -> Self {
        Self::decompose_impl(a, SLAB_WIDTH, BulkMode::Plain, scratch)
    }

    /// Forces the M4RI combination-table trailing update.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn decompose_m4ri(a: BitMatrix, scratch: &mut PleScratch) -> Self {
        Self::decompose_impl(a, SLAB_WIDTH, BulkMode::M4ri, scratch)
    }

    fn decompose_impl(
        mut mat: BitMatrix,
        panel: usize,
        mode: BulkMode,
        scratch: &mut PleScratch,
    ) -> Self {
        let (rows, cols) = (mat.rows(), mat.cols());
        let mut row_perm = Perm::identity(rows);
        let mut col_perm = Perm::identity(cols);
        let limit = rows.min(cols);
        let mut frontier = 0; // pivot frontier
        let mut wstart = 0; // leftmost column position not known dead
        while frontier < limit && wstart < cols {
            let wend = (wstart + panel).min(cols);
            let pivots = factor_panel(
                &mut mat,
                &mut row_perm,
                &mut col_perm,
                frontier,
                wstart,
                wend,
            );
            if pivots == 0 {
                // The whole window is permanently zero below the frontier.
                wstart = wend;
            } else {
                bulk_update(&mut mat, frontier, pivots, wend, mode, scratch);
                frontier += pivots;
                wstart = frontier;
            }
        }
        let rank = frontier;
        let mut row_image: Vec<usize> = (0..rows).collect();
        row_perm.apply(&mut row_image);
        let mut row_profile: Vec<usize> = row_image[..rank].to_vec();
        row_profile.sort_unstable();
        let mut col_image: Vec<usize> = (0..cols).collect();
        col_perm.apply(&mut col_image);
        let mut col_profile: Vec<usize> = col_image[..rank].to_vec();
        col_profile.sort_unstable();
        Self {
            lu: mat,
            p: row_perm,
            q: col_perm,
            rank,
            row_profile,
            col_profile,
        }
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
    pub fn into_matrix(self) -> BitMatrix {
        self.lu
    }

    pub(crate) fn lu_matrix(&self) -> &BitMatrix {
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
impl Ple {
    /// The in-place `L`/`U` storage: factors below the diagonal, `U` on and
    /// above it, in the eliminated row and column order.
    #[must_use]
    pub fn lu(&self) -> &BitMatrix {
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

/// Factors one panel: up to `wend - wstart` pivot steps starting at the
/// frontier, with the pivot search restricted to column positions
/// `[wstart, wend)`. Returns the number of pivots found; fewer than the
/// window width means every remaining column in the window is zero below the
/// frontier. Identical pivot structure to [`crate::dense`]'s `factor_panel`.
fn factor_panel(
    mat: &mut BitMatrix,
    row_perm: &mut Perm,
    col_perm: &mut Perm,
    frontier: usize,
    wstart: usize,
    wend: usize,
) -> usize {
    let rows = mat.rows();
    let limit = rows.min(mat.cols());
    let mut piv = frontier;
    while piv < limit && piv < wend && piv - frontier < wend - wstart {
        let Some((found_row, found_col)) = locate_pivot(mat, piv, wstart, wend) else {
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
        // The pivot bit at `(piv, piv)` is one; it stays set and becomes the
        // stored `L` factor, because the XOR clears only columns past `piv`.
        for row in (piv + 1)..rows {
            if mat.get(row, piv) {
                let (row_dst, row_src) = mat.two_live_rows(row, piv);
                xor_range(row_dst, row_src, piv + 1, wend);
            }
        }
        piv += 1;
    }
    piv - frontier
}

/// The first pivot in the unblocked sweep order, found word-wise (I8): the
/// leftmost column position in `[max(step, wstart), wend)` that is nonzero in
/// some row at or below `step`, and that column's first such row.
///
/// The column search ORs the candidate rows word by word and takes the
/// lowest set bit of the masked accumulator — `trailing_zeros` on the column
/// bitmap — so a zero column costs no per-row scan.
fn locate_pivot(
    mat: &BitMatrix,
    step: usize,
    wstart: usize,
    wend: usize,
) -> Option<(usize, usize)> {
    let start = step.max(wstart);
    if start >= wend {
        return None;
    }
    let (w_lo, w_hi) = (start / 64, (wend - 1) / 64);
    let rows = mat.rows();
    for w in w_lo..=w_hi {
        let mut acc = 0u64;
        for row in step..rows {
            acc |= mat.row(row)[w];
        }
        let lo = if w == w_lo { start - w * 64 } else { 0 };
        let hi = if w == w_hi { wend - w * 64 } else { 64 };
        acc &= mask_between(lo, hi);
        if acc != 0 {
            let col = w * 64 + acc.trailing_zeros() as usize;
            for row in step..rows {
                if mat.get(row, col) {
                    return Some((row, col));
                }
            }
        }
    }
    None
}

/// Zeros bits `[0, upto)` of a live-word row, leaving the rest untouched.
pub(crate) fn clear_prefix(dst: &mut [u64], upto: usize) {
    if upto == 0 {
        return;
    }
    let w = upto / 64;
    for word in &mut dst[..w] {
        *word = 0;
    }
    if w < dst.len() {
        dst[w] &= u64::MAX << (upto - w * 64);
    }
}

/// The trailing update after a panel: the triangular solve of the pivot
/// rows' right parts, then the rank-`s` update of the trailing submatrix.
/// Both cover only columns at and past `wend`. Identical structure to
/// [`crate::dense`]'s `bulk_update`.
fn bulk_update(
    mat: &mut BitMatrix,
    frontier: usize,
    pivots: usize,
    wend: usize,
    mode: BulkMode,
    scratch: &mut PleScratch,
) {
    let cols = mat.cols();
    if wend >= cols {
        return;
    }
    // U12 = L11⁻¹ · A12, in place on the pivot rows.
    for row in frontier..(frontier + pivots) {
        for src in frontier..row {
            if mat.get(row, src) {
                let (row_dst, row_src) = mat.two_live_rows(row, src);
                xor_range(row_dst, row_src, wend, cols);
            }
        }
    }
    match mode {
        BulkMode::Plain => update_trailing_plain(mat, frontier, pivots, wend),
        BulkMode::M4ri => update_trailing_m4ri(mat, frontier, pivots, wend, scratch),
    }
}

fn update_trailing_plain(mat: &mut BitMatrix, frontier: usize, pivots: usize, wend: usize) {
    let (rows, cols) = (mat.rows(), mat.cols());
    for row in (frontier + pivots)..rows {
        for src in frontier..(frontier + pivots) {
            if mat.get(row, src) {
                let (row_dst, row_src) = mat.two_live_rows(row, src);
                xor_range(row_dst, row_src, wend, cols);
            }
        }
    }
}

/// Builds all pivot-row combinations in Gray-code order, then performs one
/// table-row XOR for each trailing row. The table key is the stored `L21`
/// factor bitset, so this is byte-for-byte the same update as the plain twin.
fn update_trailing_m4ri(
    mat: &mut BitMatrix,
    frontier: usize,
    pivots: usize,
    wend: usize,
    scratch: &mut PleScratch,
) {
    let (rows, cols) = (mat.rows(), mat.cols());
    let words = mat.row_words();
    let entries = 1usize << pivots;
    scratch.table.resize(entries * words, 0);
    scratch.table.fill(0);
    let mut previous_gray = 0usize;
    for ordinal in 1..entries {
        let gray = ordinal ^ (ordinal >> 1);
        let changed = (gray ^ previous_gray).trailing_zeros() as usize;
        scratch.table.copy_within(
            previous_gray * words..(previous_gray + 1) * words,
            gray * words,
        );
        let pivot = mat.row(frontier + changed);
        xor_range(
            &mut scratch.table[gray * words..(gray + 1) * words],
            pivot,
            wend,
            cols,
        );
        previous_gray = gray;
    }
    for row in (frontier + pivots)..rows {
        let mut key = 0usize;
        for offset in 0..pivots {
            key |= usize::from(mat.get(row, frontier + offset)) << offset;
        }
        if key != 0 {
            let table_row = &scratch.table[key * words..(key + 1) * words];
            xor_range(mat.live_row_mut(row), table_row, wend, cols);
        }
    }
}

/// XORs `src` into `dst` over bit columns `[from, to)`, both live-word slices
/// of equal length. Columns outside the range are untouched; padding bits
/// stay zero because `src`'s are.
pub(crate) fn xor_range(dst: &mut [u64], src: &[u64], from: usize, to: usize) {
    if from >= to {
        return;
    }
    let w0 = from / 64;
    let w1 = (to - 1) / 64;
    if w0 == w1 {
        dst[w0] ^= src[w0] & mask_between(from - w0 * 64, to - w0 * 64);
    } else {
        dst[w0] ^= src[w0] & (u64::MAX << (from - w0 * 64));
        for w in (w0 + 1)..w1 {
            dst[w] ^= src[w];
        }
        dst[w1] ^= src[w1] & low_mask(to - w1 * 64);
    }
}

/// XORs the whole of `src` into `dst`, live word for live word.
pub(crate) fn xor_all(dst: &mut [u64], src: &[u64]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// Bits `[lo, hi)` set within a single word, `0 <= lo < hi <= 64`.
fn mask_between(lo: usize, hi: usize) -> u64 {
    (u64::MAX << lo) & low_mask(hi)
}

/// The low `n` bits set, `1 <= n <= 64`.
fn low_mask(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}
