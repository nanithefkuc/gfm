//! `Hybrid<F>` — the sparse→dense inactivation solver.
//!
//! Four stages, matching RFC 6330 §5.4.2 in structure:
//!
//! 1. **Sparse phase.** Repeatedly pivot on the lightest active row: weight one
//!    peels, weight two picks inside the largest connected component, heavier
//!    rows inactivate their surplus columns. Coefficient elimination runs
//!    eagerly (it is cheap and sparse); the payload operations are recorded.
//! 2. **Deferred application.** Payload row operations are replayed only onto
//!    the rows the answer reads — pivots and an independent set of dense-block
//!    rows — so redundant received rows are never paid for.
//! 3. **Dense phase.** The inactivated columns form a small `g`-wide block,
//!    solved by [`Ple`].
//! 4. **Back-substitution.** The inactive solution flows back through the
//!    pivot rows, in reverse pivot order.
//!
//! Correctness does not rest on the schedule: every stage is exact Gaussian
//! elimination, so the result equals a full dense `Ple` over the same system.
//! The schedule only sets how large the dense block grows.

use alloc::vec;
use alloc::vec::Vec;

use super::{DenseRow, DenseRows};
use crate::SolveError;
use crate::dense::{Matrix, Ple, PleScratch, SmallMatrix, SolveScratch};
use crate::hybrid::deferred::DeferredLog;
use crate::hybrid::schedule::Components;
use crate::hybrid::sparse::{NOT_FROZEN, Row};
macro_rules! dispatch_small_solve {
    ($solver:expr, $order:expr; $($k:literal),+ $(,)?) => {
        match $order {
            $($k => $solver.solve_small::<$k>(),)+
            _ => unreachable!("small-matrix order exceeds dispatch table"),
        }
    };
}

/// Deferred rows processed per release group. A wider group amortizes one
/// pass over every relevant pivot row's support across more rows; the
/// accumulator stays `g`-wide either way.
const RELEASE_LANES: usize = 64;

/// Inactive-ordinal sentinel for a column that is not in the dense block.
const RELEASE_NOT_INACTIVE: u32 = u32::MAX;

use fgf::field::Elem;
use fgf::{FieldKernels, ops};

/// `acc += factors`, lane by lane over the fixed release width.
///
/// The width is a compile-time constant and both operands are arrays, so
/// the loop carries no bounds check and lowers to whole-vector field
/// addition — the reason dead lanes are staged as zero rather than skipped.
#[inline]
fn add_lanes<F: FieldKernels>(
    acc: &mut [F::Elem; RELEASE_LANES],
    factors: &[F::Elem; RELEASE_LANES],
) {
    for (slot, &factor) in acc.iter_mut().zip(factors.iter()) {
        *slot = slot.add(factor);
    }
}

/// Status of a column during the sparse phase.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Col {
    Active,
    Pivoted,
    Inactive,
}

/// Counters exposed for the acceptance tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SolveStats {
    /// Columns moved into the dense block (`g`).
    pub inactivations: usize,
    /// Columns the constructor placed in the inactive set before
    /// scheduling; a subset of `inactivations`.
    pub initial_inactivations: usize,
    /// Payload (symbol) row operations performed.
    pub row_ops: usize,
    /// Rows that widened from binary to field-valued.
    pub widenings: usize,
    /// Rows released from deferral into the dense phase.
    pub deferred_rows: usize,
    /// System rank.
    pub rank: usize,
}

/// The recovered unknowns.
pub struct Solution<F: FieldKernels> {
    values: Matrix<F>,
    determined: Vec<bool>,
    rank: usize,
}

impl<F: FieldKernels> Solution<F> {
    /// The system rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Whether every unknown was determined.
    #[must_use]
    pub fn is_full_rank(&self) -> bool {
        self.rank == self.values.rows()
    }

    /// The recovered symbol for column `col`, borrowed from the solution
    /// store. Zero for an undetermined (free) column.
    #[must_use]
    pub fn value(&self, col: usize) -> &[u8] {
        self.values.row(col)
    }

    /// Whether `col` was determined.
    #[must_use]
    pub fn is_determined(&self, col: usize) -> bool {
        self.determined[col]
    }
}

impl<F: FieldKernels> core::fmt::Debug for Solution<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Solution")
            .field("cols", &self.values.rows())
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

/// A hybrid sparse→dense linear system over `cols` unknowns with a per-row
/// payload of `sym_len` bytes.
pub struct Hybrid<F: FieldKernels> {
    cols: usize,
    sym_len: usize,
    sym_cols: usize,
    rows: Vec<Row<F>>,
    rhs: Vec<u8>,
    work_rows: Vec<Row<F>>,
    work_rhs: Vec<u8>,
    col: Vec<Col>,
    inactivated_cols: Vec<u32>,
    deferred: Vec<bool>,
    release_lane: Vec<[F::Elem; RELEASE_LANES]>,
    release_mask: Vec<u64>,
    release_factors: [F::Elem; RELEASE_LANES],
    /// Substituted contributions accumulated in the inactive-column space,
    /// indexed by inactive ordinal — the working set stays `g`-wide.
    release_acc: Vec<[F::Elem; RELEASE_LANES]>,
    /// Column → position in the sorted inactive list.
    inactive_ord: Vec<u32>,
    weight_bucket: Vec<Vec<u32>>,
    bucket_pos: Vec<u32>,
    col_rows: Vec<Vec<u32>>,
    row_gen: Vec<u32>,
    generation: u32,
    deferred_rhs_ops: Vec<(u32, u32, F::Elem)>,
    edge_pair: Vec<(u32, u32)>,
    edge_dirty: Vec<bool>,
    inactivated_touch: Vec<usize>,
    release_order: Vec<usize>,
    release_chunk_start: usize,
    release_chunk_end: usize,
    alive: Vec<bool>,
    pivots: Vec<(usize, u32)>,
    inactive: Vec<u32>,
    initial_inactive: Vec<u32>,
    active_weight: Vec<usize>,
    edge_rows: Vec<usize>,
    edges: Vec<(u32, u32)>,
    active_of_pivot: Vec<u32>,
    residual: Vec<usize>,
    basis_rows: Vec<usize>,
    /// Frozen ordinal → u32 byte offset of that column's dense-block row.
    frozen_src: Vec<u32>,
    /// Frozen ordinals of the pivot row being substituted, as scratch for
    /// one gather call; reused across pivots so a solve allocates nothing
    /// after the first pivots size it.
    gather_scratch: Vec<u32>,
    needed: Vec<bool>,
    schedule: Components,
    pivot_cols: Vec<u32>,
    pivot_coeffs: Vec<F::Elem>,
    /// Snapshot of the packed pivot row's frozen words, applied to each
    /// eliminated row as one whole-word XOR.
    pivot_frozen: Vec<u64>,
    /// Lazily gathered full support (active plus frozen) of a packed pivot
    /// row, built only when a cold (widening or non-unit-factor) merge
    /// needs a flat sorted source.
    pivot_all_cols: Vec<u32>,
    pivot_gathered: bool,
    /// Column → frozen ordinal ([`NOT_FROZEN`] while active).
    frozen_ord: Vec<u32>,
    /// Frozen ordinal → column; indexes bit positions in row word vectors.
    frozen_cols: Vec<u32>,
    /// Frozen ordinal → inactive ordinal, built once the inactive set is
    /// final. Every frozen column is inactive, so the release substitution
    /// reaches its accumulator slot in one load instead of two.
    frozen_slot: Vec<u32>,
    verify_rhs: Vec<u8>,
    log: DeferredLog<F>,
    rank_ple: Option<Ple<F>>,
    solve_ple: Option<Ple<F>>,
    dense_rhs: Option<Matrix<F>>,
    x_inactive: Option<Matrix<F>>,
    dense_kernel: Option<Matrix<F>>,
    dependencies: Option<Matrix<F>>,
    ple_scratch: PleScratch<F>,
    solve_scratch: SolveScratch<F>,
}

impl<F: FieldKernels> Hybrid<F> {
    /// A system over `cols` unknowns whose payloads are `sym_len` bytes each.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] is never returned here; construction only
    /// fails via the panic path if `sym_len` is not a whole number of field
    /// elements.
    ///
    /// # Panics
    ///
    /// Panics unless `sym_len` is a multiple of `F::BYTES`.
    #[must_use]
    pub fn new(cols: usize, sym_len: usize) -> Self {
        Self::with_initial_inactive(cols, sym_len, &[])
    }

    /// A system whose columns in `initial_inactive` — sorted, distinct, in
    /// range — enter the inactive set before sparse-phase scheduling and
    /// stay there across every solve of this solver.
    ///
    /// Scheduling never pivots an inactive column in the sparse phase; the
    /// column goes straight to the dense block. This is the input
    /// permanently-inactive constructions need — RFC 6330 §5.4.2.2 starts
    /// its decoder with the last P columns already in U.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] is never returned here; construction
    /// only fails via the panic path on malformed input.
    ///
    /// # Panics
    ///
    /// Panics unless `sym_len` is a multiple of `F::BYTES` and
    /// `initial_inactive` is sorted, distinct, and inside `0..cols`.
    #[must_use]
    pub fn with_initial_inactive(cols: usize, sym_len: usize, initial_inactive: &[u32]) -> Self {
        assert!(
            sym_len.is_multiple_of(F::BYTES),
            "symbol length must be a whole number of field elements"
        );
        assert!(
            initial_inactive.windows(2).all(|pair| pair[0] < pair[1]),
            "initial inactive columns must be sorted and distinct"
        );
        assert!(
            initial_inactive.last().is_none_or(|&c| (c as usize) < cols),
            "initial inactive column out of range"
        );
        Self {
            cols,
            sym_len,
            sym_cols: sym_len / F::BYTES,
            deferred: Vec::new(),
            weight_bucket: Vec::new(),
            bucket_pos: Vec::new(),
            col_rows: Vec::new(),
            row_gen: Vec::new(),
            generation: 0,
            release_lane: Vec::new(),
            release_mask: Vec::new(),
            release_factors: [F::Elem::ZERO; RELEASE_LANES],
            release_acc: Vec::new(),
            inactive_ord: Vec::new(),
            edge_pair: Vec::new(),
            edge_dirty: Vec::new(),
            inactivated_touch: Vec::new(),
            release_order: Vec::new(),
            release_chunk_start: 0,
            release_chunk_end: 0,
            deferred_rhs_ops: Vec::new(),
            rows: Vec::new(),
            rhs: Vec::new(),
            work_rows: Vec::new(),
            work_rhs: Vec::new(),
            inactivated_cols: Vec::new(),
            col: Vec::new(),
            alive: Vec::new(),
            pivots: Vec::new(),
            inactive: Vec::new(),
            initial_inactive: initial_inactive.to_vec(),
            active_weight: Vec::new(),
            edge_rows: Vec::new(),
            edges: Vec::new(),
            active_of_pivot: Vec::new(),
            residual: Vec::new(),
            basis_rows: Vec::new(),
            needed: Vec::new(),
            schedule: Components::new(),
            frozen_src: Vec::new(),
            gather_scratch: Vec::new(),
            pivot_cols: Vec::new(),
            pivot_coeffs: Vec::new(),
            pivot_frozen: Vec::new(),
            pivot_all_cols: Vec::new(),
            pivot_gathered: false,
            frozen_ord: Vec::new(),
            frozen_cols: Vec::new(),
            frozen_slot: Vec::new(),
            verify_rhs: Vec::new(),
            log: DeferredLog::new(),
            rank_ple: None,
            solve_ple: None,
            dense_rhs: None,
            x_inactive: None,
            dense_kernel: None,
            dependencies: None,
            ple_scratch: PleScratch::new(),
            solve_scratch: SolveScratch::new(),
        }
    }

    /// Adds a binary (all-ones) equation on the sorted, distinct `support`,
    /// with `rhs` its `sym_len`-byte payload.
    ///
    /// # Panics
    ///
    /// Panics if `support` is not sorted/in range, or `rhs` is the wrong size.
    pub fn push_binary_row(&mut self, support: &[u32], rhs: &[u8]) {
        self.check_support(support);
        assert_eq!(rhs.len(), self.sym_len, "payload length");
        self.rows.push(Row::binary(support.to_vec()));
        self.rhs.extend_from_slice(rhs);
        self.deferred.push(false);
    }

    /// Adds a field-valued equation with `coeffs` parallel to `support`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are wrong or `support` is not sorted/in range.
    pub fn push_field_row(&mut self, support: &[u32], coeffs: &[F::Elem], rhs: &[u8]) {
        self.check_support(support);
        assert_eq!(support.len(), coeffs.len(), "coefficient count");
        assert_eq!(rhs.len(), self.sym_len, "payload length");
        self.rows
            .push(Row::field(support.to_vec(), coeffs.to_vec()));
        self.rhs.extend_from_slice(rhs);
        self.deferred.push(false);
    }

    /// Adds a field-valued equation that is **deferred**: excluded from
    /// sparse-phase scheduling and elimination, and released into the
    /// dense phase with its pivoted-column entries substituted out in one
    /// time-ordered pass. Dense rows (HDPC-style bands) pay `O(entries +
    /// fill)` once instead of one full-length merge per pivot they touch.
    ///
    /// Deferral changes the schedule, never the answer: rank, solution,
    /// and inconsistency verdicts are identical to pushing the same row
    /// through [`Self::push_field_row`].
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::push_field_row`].
    pub fn push_deferred_field_row(&mut self, support: &[u32], coeffs: &[F::Elem], rhs: &[u8]) {
        self.check_support(support);
        assert_eq!(support.len(), coeffs.len(), "coefficient count");
        assert_eq!(rhs.len(), self.sym_len, "payload length");
        self.rows
            .push(Row::field(support.to_vec(), coeffs.to_vec()));
        self.rhs.extend_from_slice(rhs);
        self.deferred.push(true);
    }
    /// Appends every equation exposed by `rows`, preserving source order.
    ///
    /// This is the dependency-direction seam for graph crates and codecs:
    /// they implement [`DenseRows`] for their local row store, then feed it
    /// into `gfm` without `gfm` depending on the producer.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::push_binary_row`] and
    /// [`Self::push_field_row`].
    pub fn extend_rows<R: DenseRows<F> + ?Sized>(&mut self, rows: &R) {
        rows.for_each_row(&mut |row| match row {
            DenseRow::Binary { support, rhs } => self.push_binary_row(support, rhs),
            DenseRow::Field {
                support,
                coeffs,
                rhs,
            } => self.push_field_row(support, coeffs, rhs),
        });
    }

    fn check_support(&self, support: &[u32]) {
        assert!(
            support.windows(2).all(|w| w[0] < w[1]),
            "support must be sorted and distinct"
        );
        assert!(
            support.last().is_none_or(|&c| (c as usize) < self.cols),
            "support column out of range"
        );
    }

    /// Solves the system, matching a full dense [`Ple`] on rank, solution,
    /// and inconsistency verdict.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics if the requested output geometry overflows `usize`.
    pub fn solve(&mut self) -> Result<Solution<F>, SolveError> {
        let mut values =
            Matrix::<F>::zeros(self.cols, self.sym_cols).expect("validated hybrid output geometry");
        let mut determined = vec![false; self.cols];
        let stats = self.run_into(true, &mut values, &mut determined)?;
        Ok(Solution {
            values,
            determined,
            rank: stats.rank,
        })
    }

    /// Solves into caller-owned storage. Reusing `values`, `determined`, and
    /// this solver after one warm-up solve performs no allocations.
    ///
    /// Returns the system rank. Undetermined columns are written as zero.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics unless `values` is `cols × (sym_len / F::BYTES)` and
    /// `determined.len() == cols`.
    pub fn solve_into(
        &mut self,
        values: &mut Matrix<F>,
        determined: &mut [bool],
    ) -> Result<usize, SolveError> {
        Ok(self.run_into(true, values, determined)?.rank)
    }

    /// Solves and reports the schedule/op counters. `defer` selects the
    /// deferred payload path.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics if the requested output geometry overflows `usize`.
    #[cfg(feature = "internals")]
    pub fn solve_with_stats(
        &mut self,
        defer: bool,
    ) -> Result<(Solution<F>, SolveStats), SolveError> {
        let mut values =
            Matrix::<F>::zeros(self.cols, self.sym_cols).expect("validated hybrid output geometry");
        let mut determined = vec![false; self.cols];
        let stats = self.run_into(defer, &mut values, &mut determined)?;
        Ok((
            Solution {
                values,
                determined,
                rank: stats.rank,
            },
            stats,
        ))
    }

    /// Allocation-free form of [`Self::solve_with_stats`].
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::solve_into`].
    #[cfg(feature = "internals")]
    pub fn solve_into_with_stats(
        &mut self,
        defer: bool,
        values: &mut Matrix<F>,
        determined: &mut [bool],
    ) -> Result<SolveStats, SolveError> {
        self.run_into(defer, values, determined)
    }

    fn prepare_work(&mut self) {
        let (n, m) = (self.cols, self.rows.len());
        while self.work_rows.len() < m {
            self.work_rows.push(Row::empty());
        }
        for (work, source) in self.work_rows.iter_mut().zip(&self.rows) {
            work.reset_from(source);
        }
        self.work_rows.truncate(m);
        self.work_rhs.clear();
        self.work_rhs.extend_from_slice(&self.rhs);

        self.col.clear();
        self.col.resize(n, Col::Active);
        self.alive.clear();
        self.alive.resize(m, true);
        self.deferred.resize(m, false);
        for (r, &is_deferred) in self.deferred.iter().enumerate() {
            if is_deferred {
                self.alive[r] = false;
            }
        }
        // The lane store is sized, not cleared: every read is guarded by
        // `release_mask` (which the release pass zeroes for itself) except
        // the inactive columns the emit loop walks, and the release pass
        // zeroes exactly those. An `n`-wide lane memset per solve is the
        // single largest fixed cost this phase used to carry.
        if self.release_lane.len() < n {
            self.release_lane.resize(n, [F::Elem::ZERO; RELEASE_LANES]);
        }
        self.release_lane.truncate(n);
        if self.release_mask.len() < n {
            self.release_mask.resize(n, 0u64);
        }
        self.release_mask.truncate(n);
        self.inactive_ord.clear();
        self.inactive_ord.resize(n, RELEASE_NOT_INACTIVE);
        self.deferred_rhs_ops.clear();
        self.active_weight.clear();
        self.active_weight.resize(m, 0);
        self.needed.clear();
        self.needed.resize(m, false);
        self.pivots.clear();
        self.inactive.clear();
        for &column in &self.initial_inactive {
            self.col[column as usize] = Col::Inactive;
            self.inactive.push(column);
        }
        // Frozen-ordinal space: the initially inactive columns take the
        // first ordinals, then packed work rows split their input support.
        self.assign_initial_frozen();

        // Active weights are maintained incrementally from here on: the
        // scheduling loop never recounts them. Rows change weight only
        // when a merge rewrites them (recomputed in the pivot loop) or
        // when a column they contain is inactivated (decremented then).
        // Every live row also sits in the queue for its current weight,
        // so minimum-weight selection is a queue scan, not a row scan.
        self.bucket_pos.clear();
        self.bucket_pos.resize(m, 0);
        self.seed_weight_queues();
        self.edge_pair.clear();
        self.edge_pair.resize(m, (0, 0));
        self.edge_dirty.clear();
        self.edge_dirty.resize(m, true);
        self.inactivated_cols.clear();
        self.rebuild_column_index();
        self.row_gen.clear();
        self.row_gen.resize(m, 0);
        self.generation = 0;
        self.edge_rows.clear();
        self.edges.clear();
        self.active_of_pivot.clear();
        self.residual.clear();
        self.basis_rows.clear();
        self.log.clear();

        if self.pivot_cols.capacity() < n {
            self.pivot_cols
                .reserve(n.saturating_sub(self.pivot_cols.len()));
        }
        if self.pivot_coeffs.capacity() < n {
            self.pivot_coeffs
                .reserve(n.saturating_sub(self.pivot_coeffs.len()));
        }
        self.verify_rhs.clear();
        self.verify_rhs.resize(self.sym_len, 0);
    }

    /// Puts every live row in the queue for its initial active weight.
    ///
    /// Buckets cover the weights actually reached; [`Self::bucket_move`]
    /// grows the span when a merge pushes a row past the current bound, so
    /// the initial span costs the maximum input weight, not the column
    /// count.
    fn seed_weight_queues(&mut self) {
        let bucket_count = self
            .rows
            .iter()
            .map(|row| row.cols.len() + 2)
            .max()
            .unwrap_or(2)
            .min(self.cols + 2);
        for bucket in &mut self.weight_bucket {
            bucket.clear();
        }
        if self.weight_bucket.len() < bucket_count {
            self.weight_bucket.resize_with(bucket_count, Vec::new);
        }
        for r in 0..self.rows.len() {
            if !self.alive[r] {
                continue;
            }
            let weight = self.initial_active_weight(r);
            self.active_weight[r] = weight;
            if weight > 0 {
                self.bucket_pos[r] = self.weight_bucket[weight].len() as u32;
                self.weight_bucket[weight].push(r as u32);
            }
        }
    }

    /// Rebuilds the column-to-row index over the initial supports: a
    /// pivot's elimination visits exactly the rows listed under its column
    /// (plus merge-time additions) instead of scanning every row.
    /// Cancellations leave stale entries — false positives filtered by the
    /// coefficient lookup — which keeps the index append-only.
    ///
    /// Deferred rows are left out: they never pivot and never merge, so
    /// listing them would only add probes the liveness check throws away —
    /// at max `K` that is one entry per dense HDPC coefficient, in every
    /// column.
    fn rebuild_column_index(&mut self) {
        let n = self.cols;
        self.col_rows.resize(n, Vec::new());
        self.col_rows.truncate(n);
        for column in &mut self.col_rows {
            column.clear();
        }
        for r in 0..self.rows.len() {
            if !self.alive[r] {
                continue;
            }
            for &column in &self.work_rows[r].cols {
                self.col_rows[column as usize].push(r as u32);
            }
        }
    }

    fn rhs_row(&self, r: usize) -> &[u8] {
        &self.work_rhs[r * self.sym_len..(r + 1) * self.sym_len]
    }

    /// Gives the initially inactive columns the first frozen ordinals and
    /// splits every packed work row's support accordingly: those columns
    /// leave the active lists and become bit words before any scheduling.
    fn assign_initial_frozen(&mut self) {
        self.frozen_ord.clear();
        self.frozen_ord.resize(self.cols, NOT_FROZEN);
        self.frozen_cols.clear();
        for (ordinal, &column) in self.initial_inactive.iter().enumerate() {
            self.frozen_ord[column as usize] = ordinal as u32;
            self.frozen_cols.push(column);
        }
        if self.frozen_cols.is_empty() {
            return;
        }
        let columns = &self.frozen_cols;
        let ordinals = &self.frozen_ord;
        for work in &mut self.work_rows {
            if work.binary {
                work.freeze_from_list(columns, ordinals);
            }
        }
    }

    /// The row's active weight at prepare time: a packed row's list is
    /// exactly its active support after the split; a widened row keeps
    /// frozen entries in the list and counts active ones directly.
    fn initial_active_weight(&self, r: usize) -> usize {
        if self.work_rows[r].binary {
            self.work_rows[r].cols.len()
        } else {
            self.work_rows[r]
                .cols
                .iter()
                .filter(|&&c| self.col[c as usize] == Col::Active)
                .count()
        }
    }

    /// `rhs[dst] ^= factor · rhs[src]`.
    fn rhs_axpy(&mut self, dst: usize, src: usize, factor: F::Elem) {
        debug_assert_ne!(dst, src);
        let sym = self.sym_len;
        let (lo, hi) = if dst < src { (dst, src) } else { (src, dst) };
        let (head, tail) = self.work_rhs.split_at_mut(hi * sym);
        let (dst_s, src_s) = if dst < src {
            (&mut head[lo * sym..lo * sym + sym], &tail[..sym])
        } else {
            (&mut tail[..sym], &head[lo * sym..lo * sym + sym])
        };
        crate::row_ops::mul_add::<F>(dst_s, factor, src_s);
    }

    fn prepare_ple<'a>(
        slot: &'a mut Option<Ple<F>>,
        rows: usize,
        cols: usize,
        scratch: &mut PleScratch<F>,
    ) -> &'a mut Ple<F> {
        let wrong_geometry = slot
            .as_ref()
            .is_none_or(|ple| ple.rows() != rows || ple.cols() != cols);
        if wrong_geometry {
            let matrix = Matrix::<F>::zeros(rows, cols).expect("validated hybrid dense geometry");
            *slot = Some(Ple::decompose(matrix, scratch));
        }
        let ple = slot.as_mut().expect("initialized above");
        for row in 0..rows {
            ple.matrix_for_redecomposition().row_mut(row).fill(0);
        }
        ple
    }

    fn prepare_matrix(slot: &mut Option<Matrix<F>>, rows: usize, cols: usize) -> &mut Matrix<F> {
        let wrong_geometry = slot
            .as_ref()
            .is_none_or(|matrix| matrix.rows() != rows || matrix.cols() != cols);
        if wrong_geometry {
            *slot = Some(Matrix::<F>::zeros(rows, cols).expect("validated hybrid dense geometry"));
        }
        let matrix = slot.as_mut().expect("initialized above");
        for row in 0..rows {
            matrix.row_mut(row).fill(0);
        }
        matrix
    }
    fn solve_small<const K: usize>(&mut self) -> Result<(), SolveError> {
        debug_assert_eq!(self.inactive.len(), K);
        debug_assert_eq!(self.basis_rows.len(), K);
        let mut coefficients = SmallMatrix::<F, K>::zeros();
        for (row_index, &source_row) in self.basis_rows.iter().enumerate() {
            for (col_index, &column) in self.inactive.iter().enumerate() {
                let ordinal = self.frozen_ord[column as usize];
                coefficients.set(
                    row_index,
                    col_index,
                    self.work_rows[source_row].get(column, ordinal),
                );
            }
        }
        coefficients.solve_into(
            self.dense_rhs.as_ref().expect("prepared"),
            self.x_inactive.as_mut().expect("prepared"),
        )
    }

    #[allow(clippy::too_many_lines, clippy::needless_range_loop)]
    fn run_into(
        &mut self,
        defer: bool,
        values: &mut Matrix<F>,
        determined: &mut [bool],
    ) -> Result<SolveStats, SolveError> {
        let (n, m) = (self.cols, self.rows.len());
        assert_eq!(values.rows(), n, "solution row count");
        assert_eq!(values.cols(), self.sym_cols, "solution column count");
        assert_eq!(determined.len(), n, "determinedness length");
        self.prepare_work();
        for row in 0..n {
            values.row_mut(row).fill(0);
        }
        determined.fill(false);
        let mut stats = SolveStats::default();

        // Sparse phase. Inactivation is a separate step: once a weight-r row
        // is chosen, r-1 columns leave the active system, making that row a
        // weight-one pivot on the next iteration. Active weights are
        // maintained incrementally (initialized in `prepare_work`,
        // recomputed on merge, decremented on inactivation) and never
        // recounted here.
        loop {
            let mut min_weight = 0usize;
            for weight in 1..self.weight_bucket.len() {
                if !self.weight_bucket[weight].is_empty() {
                    min_weight = weight;
                    break;
                }
            }
            if min_weight == 0 {
                break;
            }

            let pivot_row = if min_weight == 2 {
                self.edge_rows.clear();
                self.edges.clear();
                for &r in &self.weight_bucket[2] {
                    let r = r as usize;
                    // Edges are cached per row and re-extracted only when
                    // the row's support or weight changed since its last
                    // use — most weight-two rows are stable across
                    // consecutive peeling steps.
                    if self.edge_dirty[r] {
                        let mut found = [0u32; 2];
                        let mut seen = 0;
                        for &c in &self.work_rows[r].cols {
                            if self.col[c as usize] == Col::Active {
                                found[seen] = c;
                                seen += 1;
                                if seen == 2 {
                                    break;
                                }
                            }
                        }
                        self.edge_pair[r] = (found[0], found[1]);
                        self.edge_dirty[r] = false;
                    }
                    self.edge_rows.push(r);
                    self.edges.push(self.edge_pair[r]);
                }
                let edge = self
                    .schedule
                    .largest_component_edge(&self.edges, n)
                    .expect("a weight-two row exists");
                self.edge_rows[edge]
            } else {
                self.weight_bucket[min_weight][0] as usize
            };

            self.active_of_pivot.clear();
            self.active_of_pivot.extend(
                self.work_rows[pivot_row]
                    .cols
                    .iter()
                    .copied()
                    .filter(|&c| self.col[c as usize] == Col::Active),
            );
            if min_weight > 1 {
                self.inactivated_cols.clear();
                self.inactivated_cols
                    .extend_from_slice(&self.active_of_pivot[1..]);
                for &column in &self.inactivated_cols {
                    self.col[column as usize] = Col::Inactive;
                    self.inactive.push(column);
                }
                self.weights_on_inactivated();
                continue;
            }

            let pivot_col = self.active_of_pivot[0];
            self.col[pivot_col as usize] = Col::Pivoted;
            self.alive[pivot_row] = false;
            self.bucket_move(pivot_row, 0);
            self.pivots.push((pivot_row, pivot_col));

            self.pivot_cols.clear();
            self.pivot_coeffs.clear();
            self.pivot_frozen.clear();
            self.pivot_gathered = false;
            self.pivot_cols
                .extend_from_slice(&self.work_rows[pivot_row].cols);
            if self.work_rows[pivot_row].binary {
                self.pivot_frozen
                    .extend_from_slice(&self.work_rows[pivot_row].frozen);
            } else {
                self.pivot_coeffs
                    .extend_from_slice(&self.work_rows[pivot_row].coeffs);
            }
            let pivot_binary = self.work_rows[pivot_row].binary;
            let pivot_inv = self.work_rows[pivot_row].get(pivot_col, NOT_FROZEN).inv();
            // A packed pivot whose active list is the pivot column alone
            // takes the fused path: its destinations need one search, not a
            // full support merge. Weight-one selection makes this the shape
            // of nearly every pivot in a peeling schedule.
            let singleton = pivot_binary && self.pivot_cols.len() == 1;
            debug_assert!(!singleton || self.pivot_cols[0] == pivot_col);

            self.generation += 1;
            let generation = self.generation;
            let column = pivot_col as usize;
            let listed = self.col_rows[column].len();
            for k in 0..listed {
                let r = self.col_rows[column][k] as usize;
                if self.row_gen[r] == generation {
                    continue; // duplicate listing (re-added entry)
                }
                self.row_gen[r] = generation;
                if !self.alive[r] {
                    continue;
                }
                let factor;
                let merged_weight;
                if singleton && self.work_rows[r].binary {
                    if !self.work_rows[r].eliminate_singleton(pivot_col, &self.pivot_frozen) {
                        continue; // stale listing (cancelled entry)
                    }
                    factor = F::Elem::ONE;
                    merged_weight = self.work_rows[r].cols.len();
                } else {
                    let entry = self.work_rows[r].get(pivot_col, NOT_FROZEN);
                    if entry.is_zero() {
                        continue; // stale listing (cancelled entry)
                    }
                    factor = entry.mul(pivot_inv);
                    merged_weight = if self.work_rows[r].binary && pivot_binary && factor.is_one() {
                        // The hot GF(2) combination: active lists XOR, frozen
                        // words XOR wholesale.
                        self.work_rows[r].axpy_xor_parts(
                            &self.pivot_cols,
                            &self.pivot_frozen,
                            |added| {
                                // The merge widened this row's support; index
                                // the new columns so their future pivots find
                                // it.
                                self.col_rows[added as usize].push(r as u32);
                            },
                            |c| self.col[c as usize] == Col::Active,
                        )
                    } else {
                        // Cold path: a widened destination or a non-unit
                        // factor. A packed source needs its full sorted
                        // support flat.
                        if pivot_binary && !self.pivot_gathered {
                            self.gather_pivot_all_cols();
                            self.pivot_gathered = true;
                        }
                        if self.work_rows[r].binary
                            && self.work_rows[r].materialize(&self.frozen_cols)
                        {
                            stats.widenings += 1;
                        }
                        let src_cols = if pivot_binary {
                            &self.pivot_all_cols
                        } else {
                            &self.pivot_cols
                        };
                        let (_, weight) = self.work_rows[r].axpy_coeffs_slices(
                            factor,
                            src_cols,
                            &self.pivot_coeffs,
                            pivot_binary,
                            |added| {
                                self.col_rows[added as usize].push(r as u32);
                            },
                            |c| self.col[c as usize] == Col::Active,
                        );
                        weight
                    };
                }
                // The merge rewrote the row's support: its cached edge
                // (if any) is stale even when the active weight is not.
                self.edge_dirty[r] = true;
                self.bucket_move(r, merged_weight);
                self.log.record(r as u32, pivot_row as u32, factor);
                if !defer {
                    self.rhs_axpy(r, pivot_row, factor);
                    stats.row_ops += 1;
                }
            }
        }

        for (column, state) in self.col.iter_mut().enumerate() {
            if *state == Col::Active {
                *state = Col::Inactive;
                self.inactive.push(column as u32);
            }
        }
        self.inactive.sort_unstable();
        for (ordinal, &column) in self.inactive.iter().enumerate() {
            self.inactive_ord[column as usize] = ordinal as u32;
        }
        // Frozen ordinals name inactivated columns, so their ordinal space
        // maps straight into the dense block. Both the release substitution
        // and back-substitution read pivot supports through this table
        // instead of the column round trip.
        self.frozen_slot.clear();
        self.frozen_slot.reserve(self.frozen_cols.len());
        for &column in &self.frozen_cols {
            let slot = self.inactive_ord[column as usize];
            debug_assert_ne!(slot, RELEASE_NOT_INACTIVE, "a frozen column is inactive");
            self.frozen_slot.push(slot);
        }
        stats.inactivations = self.inactive.len();
        stats.initial_inactivations = self.initial_inactive.len();
        stats.deferred_rows = self.deferred.iter().filter(|&&d| d).count();
        // Release deferred rows into the residual: substitute out their
        // pivoted-column entries in pivot-time order (their right-hand
        // sides are applied after the deferred payload replay below).
        self.release_deferred();
        self.residual.extend((0..m).filter(|&r| self.alive[r]));

        // Rank the residual coefficient block first. Its row profile is the
        // exact independent subset whose payloads the dense solve needs.
        let g = self.inactive.len();
        {
            let rank_ple = Self::prepare_ple(
                &mut self.rank_ple,
                self.residual.len(),
                g,
                &mut self.ple_scratch,
            );
            let matrix = rank_ple.matrix_for_redecomposition();
            for (i, &row) in self.residual.iter().enumerate() {
                for (j, &column) in self.inactive.iter().enumerate() {
                    let ordinal = self.frozen_ord[column as usize];
                    let coefficient = self.work_rows[row].get(column, ordinal);
                    if !coefficient.is_zero() {
                        matrix.set(i, j, coefficient);
                    }
                }
            }
            rank_ple.redecompose(&mut self.ple_scratch);
        }
        let dense_rank = self.rank_ple.as_ref().expect("prepared").rank();
        self.basis_rows.extend(
            self.rank_ple
                .as_ref()
                .expect("prepared")
                .row_rank_profile()
                .iter()
                .map(|&i| self.residual[i]),
        );

        for &(row, _) in &self.pivots {
            self.needed[row] = true;
        }
        for &row in &self.basis_rows {
            self.needed[row] = true;
        }
        if defer {
            for index in 0..self.log.ops().len() {
                let op = self.log.ops()[index];
                if self.needed[op.dst as usize] {
                    self.rhs_axpy(op.dst as usize, op.src as usize, op.factor);
                    stats.row_ops += 1;
                }
            }
        }
        // Deferred rows' right-hand sides combine the (now final) pivot
        // row payloads with the factors collected at release.
        for index in 0..self.deferred_rhs_ops.len() {
            let (dst, src, factor) = self.deferred_rhs_ops[index];
            self.rhs_axpy(dst as usize, src as usize, factor);
            stats.row_ops += 1;
        }

        // Solve only the independent residual rows over the inactive columns.
        // The compact path wins through its maximum supported order in three
        // pinned runs; the general decomposition handles wider, deficient, or
        // rectangular blocks.
        let use_small = F::BYTES == 1 && g <= 64 && self.basis_rows.len() == g;
        if !use_small {
            let solve_ple = Self::prepare_ple(
                &mut self.solve_ple,
                self.basis_rows.len(),
                g,
                &mut self.ple_scratch,
            );
            let matrix = solve_ple.matrix_for_redecomposition();
            for (i, &row) in self.basis_rows.iter().enumerate() {
                for (j, &column) in self.inactive.iter().enumerate() {
                    let ordinal = self.frozen_ord[column as usize];
                    let coefficient = self.work_rows[row].get(column, ordinal);
                    if !coefficient.is_zero() {
                        matrix.set(i, j, coefficient);
                    }
                }
            }
            solve_ple.redecompose(&mut self.ple_scratch);
            debug_assert_eq!(solve_ple.rank(), dense_rank);
        }
        {
            let dense_rhs =
                Self::prepare_matrix(&mut self.dense_rhs, self.basis_rows.len(), self.sym_cols);
            for (i, &row) in self.basis_rows.iter().enumerate() {
                let start = row * self.sym_len;
                dense_rhs
                    .row_mut(i)
                    .copy_from_slice(&self.work_rhs[start..start + self.sym_len]);
            }
        }
        Self::prepare_matrix(&mut self.x_inactive, g, self.sym_cols);
        if use_small {
            dispatch_small_solve!(self, g;
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                62, 63, 64
            )?;
        } else {
            self.solve_ple.as_ref().expect("prepared").solve_into(
                self.dense_rhs.as_ref().expect("prepared"),
                self.x_inactive.as_mut().expect("prepared"),
                &mut self.solve_scratch,
            )?;
        }

        let x_inactive = self.x_inactive.as_ref().expect("prepared");
        for (j, &column) in self.inactive.iter().enumerate() {
            values
                .row_mut(column as usize)
                .copy_from_slice(x_inactive.row(j));
        }
        // Frozen ordinal → u32 byte offset of that column's dense-block row.
        // Back-substitution resolves millions of sources through this, so
        // the row-map lookup and pitch multiply are paid once per column
        // instead of once per entry, and the gather kernel reads the
        // 4-byte table directly instead of staged fat pointers.
        let mut frozen_src = core::mem::take(&mut self.frozen_src);
        frozen_src.clear();
        frozen_src.reserve(self.frozen_slot.len());
        for &slot in &self.frozen_slot {
            frozen_src.push(
                u32::try_from(x_inactive.row_offset(slot as usize))
                    .expect("dense-block region offset within 4 GiB"),
            );
        }
        self.frozen_src = frozen_src;

        // The gather folds the packed rows' frozen supports — every
        // coefficient one, so the fold is a raw XOR with the destination
        // row held in vector registers across the whole support. The
        // ordinal scratch is a persistent field: a solve allocates nothing
        // once its first pivots have sized it.
        let dense_region = self.x_inactive.as_ref().expect("prepared").region();
        for &(row, pivot_col) in self.pivots.iter().rev() {
            values
                .row_mut(pivot_col as usize)
                .copy_from_slice(self.rhs_row(row));
            let work_row = &self.work_rows[row];
            if work_row.binary {
                // A packed pivot row carries unit coefficients, its pivot
                // column, and frozen bits over inactivated columns only —
                // every one of which is a row of the dense block, already
                // final. Reading them there instead of back through `values`
                // costs one table lookup per entry and needs no aliasing
                // split: the two matrices are disjoint.
                let offsets = &self.frozen_src;
                let dst = values.row_mut(pivot_col as usize);
                let mut gather = core::mem::take(&mut self.gather_scratch);
                gather.clear();
                work_row.for_each_frozen_ordinal(|ordinal| {
                    gather.push(offsets[ordinal]);
                });
                ops::add_gather::<F>(dense_region, dst, &gather);
                self.gather_scratch = gather;
                for &column in &work_row.cols {
                    if column != pivot_col {
                        let (dst, src) = values.two_live_rows(pivot_col as usize, column as usize);
                        ops::add_assign::<F>(dst, src);
                    }
                }
            } else {
                let frozen_cols = &self.frozen_cols;
                work_row.for_each_entry(frozen_cols, |column, coefficient| {
                    if column == pivot_col {
                        return;
                    }
                    let (dst, src) = values.two_live_rows(pivot_col as usize, column as usize);
                    crate::row_ops::mul_add::<F>(dst, coefficient, src);
                });
            }
            let ordinal = self.frozen_ord[pivot_col as usize];
            let pivot = self.work_rows[row].get(pivot_col, ordinal);
            if !pivot.is_one() {
                ops::mul_assign::<F>(values.row_mut(pivot_col as usize), pivot.inv());
            }
        }

        // A column is determined exactly when every vector in the full right
        // kernel is zero there. Lift the inactive block's kernel through the
        // same back-substitution as the payload solution; this preserves
        // cancellations that a per-dependency boolean walk would miss.
        let free = g - dense_rank;
        if free == 0 {
            determined.fill(true);
        } else {
            Self::prepare_matrix(&mut self.dense_kernel, g, free);
            self.solve_ple
                .as_ref()
                .expect("prepared")
                .kernel_into(self.dense_kernel.as_mut().expect("prepared"));
            Self::prepare_matrix(&mut self.dependencies, n, free);
            {
                let dense_kernel = self.dense_kernel.as_ref().expect("prepared");
                let dependencies = self.dependencies.as_mut().expect("prepared");
                for (j, &column) in self.inactive.iter().enumerate() {
                    dependencies
                        .row_mut(column as usize)
                        .copy_from_slice(dense_kernel.row(j));
                }
                for &(row, pivot_col) in self.pivots.iter().rev() {
                    {
                        let work_row = &self.work_rows[row];
                        let frozen_cols = &self.frozen_cols;
                        work_row.for_each_entry(frozen_cols, |column, coefficient| {
                            if column == pivot_col {
                                return;
                            }
                            let (dst, src) =
                                dependencies.two_live_rows(pivot_col as usize, column as usize);
                            crate::row_ops::mul_add::<F>(dst, coefficient, src);
                        });
                    }
                    let ordinal = self.frozen_ord[pivot_col as usize];
                    let pivot = self.work_rows[row].get(pivot_col, ordinal);
                    if !pivot.is_one() {
                        ops::mul_assign::<F>(dependencies.row_mut(pivot_col as usize), pivot.inv());
                    }
                }
                #[cfg(debug_assertions)]
                for source in &self.rows {
                    for basis in 0..free {
                        let mut sum = F::Elem::ZERO;
                        for &column in &source.cols {
                            let coefficient = source.get_input(column);
                            sum =
                                sum.add(coefficient.mul(dependencies.get(column as usize, basis)));
                        }
                        debug_assert!(sum.is_zero(), "lifted dependency is not in the kernel");
                    }
                }
                for (column, is_determined) in determined.iter_mut().enumerate() {
                    *is_determined = dependencies.row(column).iter().all(|&byte| byte == 0);
                }
            }
        }

        // Pivot and dense-basis equations are satisfied by construction. Only
        // rows outside that independent set can contradict the assembled
        // answer. Each such row is checked in the cheapest form that is
        // algebraically equivalent to its input equation:
        //
        // * a released deferred row already carries its fully substituted
        //   coefficients over the inactive columns, and its payload absorbed
        //   the pivot factors above — the reduced equation holds exactly when
        //   the original one does;
        // * a binary row's coefficients are all units, so evaluation is one
        //   XOR of the selected value rows;
        // * anything else evaluates entry by entry over the input support.
        for (row, source) in self.rows.iter().enumerate() {
            if self.needed[row] {
                continue;
            }
            self.verify_rhs.fill(0);
            if self.deferred[row] {
                let work = &self.work_rows[row];
                let frozen_cols = &self.frozen_cols;
                work.for_each_entry(frozen_cols, |column, coefficient| {
                    crate::row_ops::mul_add::<F>(
                        &mut self.verify_rhs,
                        coefficient,
                        values.row(column as usize),
                    );
                });
                let expected = &self.work_rhs[row * self.sym_len..(row + 1) * self.sym_len];
                if self.verify_rhs != expected {
                    return Err(SolveError::Inconsistent { row });
                }
                continue;
            }
            if source.binary {
                for &column in &source.cols {
                    ops::add_assign::<F>(&mut self.verify_rhs, values.row(column as usize));
                }
            } else {
                for &column in &source.cols {
                    let coefficient = source.get_input(column);
                    crate::row_ops::mul_add::<F>(
                        &mut self.verify_rhs,
                        coefficient,
                        values.row(column as usize),
                    );
                }
            }
            let expected = &self.rhs[row * self.sym_len..(row + 1) * self.sym_len];
            if self.verify_rhs != expected {
                return Err(SolveError::Inconsistent { row });
            }
        }

        for (work, source) in self.work_rows.iter_mut().zip(&self.rows) {
            work.reserve_like(source);
            if work.frozen.capacity() < self.frozen_cols.len() {
                work.frozen
                    .reserve(self.frozen_cols.len().saturating_sub(work.frozen.len()));
            }
        }
        stats.rank = self.pivots.len() + dense_rank;
        Ok(stats)
    }

    /// Freezes the just-inactivated columns (`self.inactivated_cols`): each
    /// takes the next frozen ordinal, and every live row containing one
    /// moves those entries out of its active list and into its bit words.
    /// Inactivation is rare (the schedule inactivates only fill-producing
    /// columns), so a scan with a support lookup per row keeps the
    /// incremental-weight invariant without touching unaffected rows. From
    /// here on, merges carry these columns as whole-word XORs instead of
    /// per-entry list walks.
    fn weights_on_inactivated(&mut self) {
        // The column-to-row index names the candidate rows; stale
        // entries (cancelled coefficients) are filtered by the lookup.
        // A generation guard deduplicates repeated listings without
        // sorting; each affected row is then migrated once.
        for &column in &self.inactivated_cols {
            debug_assert_eq!(self.frozen_ord[column as usize], NOT_FROZEN);
            self.frozen_ord[column as usize] = self.frozen_cols.len() as u32;
            self.frozen_cols.push(column);
        }
        self.generation += 1;
        let generation = self.generation;
        let mut seen = core::mem::take(&mut self.inactivated_touch);
        seen.clear();
        for &column in &self.inactivated_cols {
            let listed = self.col_rows[column as usize].len();
            for k in 0..listed {
                let r = self.col_rows[column as usize][k] as usize;
                if self.row_gen[r] == generation || !self.alive[r] {
                    continue;
                }
                // Containment in the *active* list: frozen bits are not
                // consulted because the entries have not moved yet.
                if !self.work_rows[r].contains_in_list(column) {
                    continue;
                }
                self.row_gen[r] = generation;
                seen.push(r);
            }
        }
        for &r in &seen {
            let lost = if self.work_rows[r].binary {
                let columns = &self.inactivated_cols;
                let ordinals = &self.frozen_ord;
                self.work_rows[r].freeze_from_list(columns, ordinals)
            } else {
                // Widened rows keep entries in place (bits cannot carry a
                // coefficient); only the active weight drops.
                self.inactivated_cols
                    .iter()
                    .filter(|&&column| self.work_rows[r].contains_in_list(column))
                    .count()
            };
            self.bucket_move(r, self.active_weight[r] - lost);
        }
        self.inactivated_touch = seen;
    }

    /// Flattens a packed pivot row's snapshot into one sorted column list
    /// (active entries plus frozen bits) for the cold list-merge path.
    fn gather_pivot_all_cols(&mut self) {
        self.pivot_all_cols.clear();
        self.pivot_all_cols.extend_from_slice(&self.pivot_cols);
        for (word, &bits) in self.pivot_frozen.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                self.pivot_all_cols
                    .push(self.frozen_cols[word * 64 + bit as usize]);
            }
        }
        self.pivot_all_cols.sort_unstable();
    }

    /// Moves row `r` into the queue for `new_weight` (0 parks it). The
    /// queues hold every live row with positive active weight, exactly.
    fn bucket_move(&mut self, r: usize, new_weight: usize) {
        let old = self.active_weight[r];
        if old == new_weight {
            return;
        }
        if old > 0 && old < self.weight_bucket.len() {
            let bucket = &mut self.weight_bucket[old];
            let pos = self.bucket_pos[r] as usize;
            let last = bucket.len() - 1;
            bucket.swap(pos, last);
            self.bucket_pos[bucket[pos] as usize] = pos as u32;
            bucket.pop();
        }
        self.active_weight[r] = new_weight;
        if new_weight > 0 {
            if new_weight == 2 {
                self.edge_dirty[r] = true;
            }
            if new_weight >= self.weight_bucket.len() {
                // A merge widened the row past every input weight; the
                // queue span follows it.
                self.weight_bucket
                    .resize_with(new_weight + 1, alloc::vec::Vec::new);
            }
            let bucket = &mut self.weight_bucket[new_weight];
            self.bucket_pos[r] = bucket.len() as u32;
            bucket.push(r as u32);
        }
    }

    /// Releases every deferred row into the dense phase.
    ///
    /// Each deferred row's pivoted-column entries are substituted out in
    /// pivot-time order: popping the earliest pivot whose column still
    /// carries a (possibly evolved) coefficient and merging that frozen
    /// pivot row's entries into a column-indexed accumulator. A pivot row
    /// frozen at time `t` has zeros on all pivots with time `< t`, so the
    /// time-ordered replay reproduces exactly the coefficients eager
    /// elimination would have produced — one sparse pass instead of one
    /// full-length merge per pivot the row touches. Inactive-column
    /// coefficients become the released row's dense-phase support; the
    /// `(dst, src, factor)` right-hand-side contributions are collected
    /// for application after the deferred payload replay.
    fn release_deferred(&mut self) {
        self.release_order.clear();
        self.release_order.extend(
            self.deferred
                .iter()
                .enumerate()
                .filter(|&(_, &d)| d)
                .map(|(r, _)| r),
        );
        if self.release_order.is_empty() {
            return;
        }
        self.release_acc.clear();
        self.release_acc
            .resize(self.inactive.len(), [F::Elem::ZERO; RELEASE_LANES]);
        // Lanes are processed in groups of the accumulator width; a wider
        // dense band simply takes more groups.
        for start in (0..self.release_order.len()).step_by(RELEASE_LANES) {
            let end = (start + RELEASE_LANES).min(self.release_order.len());
            self.release_chunk_start = start;
            self.release_chunk_end = end;
            self.release_deferred_chunk();
        }
    }

    /// The batched substitution for one group of deferred rows.
    ///
    /// Substitutions write only to inactive columns — pivot rows carry
    /// nothing on pivoted columns, so a pivot column's seeded coefficient
    /// is untouched until its own pivot fires and its factor can be read
    /// straight from the seed. That makes the substitution sums order-free
    /// per column: contributions land in a `g`-wide accumulator indexed by
    /// inactive ordinal instead of an `n`-wide lane store, and each
    /// triggered pivot walks its support once for the whole group with the
    /// live lanes hoisted out of the inner loop.
    fn release_deferred_chunk(&mut self) {
        let mut lanes = [0usize; RELEASE_LANES];
        let lane_count = self.seed_release_lanes(&mut lanes);
        self.substitute_release_lanes(&lanes, lane_count);
        self.emit_released_rows(&lanes, lane_count);
    }

    /// Seeds one lane per deferred row in the group with that row's
    /// coefficients, and records which columns each lane occupies.
    /// Returns the number of lanes seeded.
    fn seed_release_lanes(&mut self, lanes: &mut [usize; RELEASE_LANES]) -> usize {
        // Only the inactive columns are read unconditionally (by the emit
        // pass); every other read is gated on `release_mask`, which is set
        // exactly where a lane was seeded. Zeroing the `g` inactive slots
        // instead of all `n` keeps this off the memset budget.
        for &column in &self.inactive {
            self.release_lane[column as usize] = [F::Elem::ZERO; RELEASE_LANES];
        }
        self.release_mask.fill(0);
        let mut lane_count = 0usize;
        for index in self.release_chunk_start..self.release_chunk_end {
            let r = self.release_order[index];
            self.alive[r] = true;
            lanes[lane_count] = r;
            let lane = lane_count;
            lane_count += 1;
            for index in 0..self.work_rows[r].cols.len() {
                let column = self.work_rows[r].cols[index];
                let coefficient = self.work_rows[r].coeffs[index];
                self.release_lane[column as usize][lane] = coefficient;
                self.release_mask[column as usize] |= 1u64 << lane;
            }
        }
        lane_count
    }

    /// Substitutes every pivot the group's lanes still touch, in pivot
    /// order, into the inactive-space accumulator.
    fn substitute_release_lanes(&mut self, lanes: &[usize; RELEASE_LANES], deferred_count: usize) {
        let mut live_lanes = [0usize; RELEASE_LANES];
        for &(pivot_row, pivot_col) in &self.pivots {
            let mask = self.release_mask[pivot_col as usize];
            if mask == 0 {
                continue;
            }
            let pivot_value = self.work_rows[pivot_row].get(pivot_col, NOT_FROZEN);
            let pivot_inv = pivot_value.inv();
            // Factors are staged over the full lane width with zeros on the
            // dead lanes: a zero factor contributes nothing, so a packed
            // pivot's substitution becomes one whole-vector addition per
            // support entry instead of one scalar multiply per live lane.
            self.release_factors = [F::Elem::ZERO; RELEASE_LANES];
            let mut live_count = 0usize;
            let mut lane = 0;
            while lane < deferred_count {
                if mask & (1u64 << lane) != 0 {
                    let factor = self.release_lane[pivot_col as usize][lane].mul(pivot_inv);
                    if !factor.is_zero() {
                        self.deferred_rhs_ops
                            .push((lanes[lane] as u32, pivot_row as u32, factor));
                        self.release_factors[lane] = factor;
                        live_lanes[live_count] = lane;
                        live_count += 1;
                    }
                }
                lane += 1;
            }
            if live_count == 0 {
                self.release_mask[pivot_col as usize] = 0;
                continue;
            }
            {
                let pivot = &self.work_rows[pivot_row];
                let ordinals = &self.inactive_ord;
                let factors = &self.release_factors;
                let acc = &mut self.release_acc;
                if pivot.binary {
                    // Every coefficient is one, so `factor · value` is the
                    // factor itself and the lane loop collapses to a
                    // fixed-width vector add the compiler keeps in registers.
                    let slots = &self.frozen_slot;
                    for &column in &pivot.cols {
                        if column == pivot_col {
                            continue; // already substituted out above
                        }
                        let slot = ordinals[column as usize];
                        debug_assert_ne!(
                            slot, RELEASE_NOT_INACTIVE,
                            "pivot rows carry nothing outside their pivot column and the inactive set"
                        );
                        if slot != RELEASE_NOT_INACTIVE {
                            add_lanes::<F>(&mut acc[slot as usize], factors);
                        }
                    }
                    pivot.for_each_frozen_ordinal(|ordinal| {
                        let slot = slots[ordinal] as usize;
                        add_lanes::<F>(&mut acc[slot], factors);
                    });
                } else {
                    let frozen_cols = &self.frozen_cols;
                    pivot.for_each_entry(frozen_cols, |column, value| {
                        let column = column as usize;
                        if column == pivot_col as usize {
                            return; // already substituted out above
                        }
                        let slot = ordinals[column];
                        debug_assert_ne!(
                            slot, RELEASE_NOT_INACTIVE,
                            "pivot rows carry nothing outside their pivot column and the inactive set"
                        );
                        if slot == RELEASE_NOT_INACTIVE {
                            return;
                        }
                        let slot = slot as usize;
                        for &lane in &live_lanes[..live_count] {
                            let updated = acc[slot][lane].add(factors[lane].mul(value));
                            acc[slot][lane] = updated;
                        }
                    });
                }
            }
            // The pivot column is fully substituted out for these lanes.
            self.release_mask[pivot_col as usize] = 0;
        }
    }

    /// Emits each released row over the (sorted) inactive columns: the
    /// seeded coefficient plus everything substituted onto it.
    fn emit_released_rows(&mut self, lanes: &[usize; RELEASE_LANES], lane_count: usize) {
        for (lane, &lane_row) in lanes.iter().enumerate().take(lane_count) {
            let work = &mut self.work_rows[lane_row];
            work.cols.clear();
            work.coeffs.clear();
            work.frozen.clear();
            for (ordinal, &column) in self.inactive.iter().enumerate() {
                let value =
                    self.release_lane[column as usize][lane].add(self.release_acc[ordinal][lane]);
                if !value.is_zero() {
                    work.cols.push(column);
                    work.coeffs.push(value);
                }
            }
            work.binary = false;
        }
    }
}

impl<F: FieldKernels> core::fmt::Debug for Hybrid<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Hybrid")
            .field("cols", &self.cols)
            .field("rows", &self.rows.len())
            .field("sym_len", &self.sym_len)
            .finish_non_exhaustive()
    }
}
