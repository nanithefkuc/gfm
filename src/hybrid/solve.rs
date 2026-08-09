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
use crate::hybrid::schedule::largest_component_edge;
use crate::hybrid::sparse::Row;
macro_rules! dispatch_small_solve {
    ($solver:expr, $order:expr; $($k:literal),+ $(,)?) => {
        match $order {
            $($k => $solver.solve_small::<$k>(),)+
            _ => unreachable!("small-matrix order exceeds dispatch table"),
        }
    };
}

use fgf::field::Elem;
use fgf::{FieldKernels, ops};

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
    /// Payload (symbol) row operations performed.
    pub row_ops: usize,
    /// Rows that widened from binary to field-valued.
    pub widenings: usize,
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
    alive: Vec<bool>,
    pivots: Vec<(usize, u32)>,
    inactive: Vec<u32>,
    active_weight: Vec<usize>,
    edge_rows: Vec<usize>,
    edges: Vec<(u32, u32)>,
    active_of_pivot: Vec<u32>,
    residual: Vec<usize>,
    basis_rows: Vec<usize>,
    needed: Vec<bool>,
    schedule_parent: Vec<usize>,
    schedule_rank: Vec<u8>,
    schedule_size: Vec<usize>,
    pivot_cols: Vec<u32>,
    pivot_coeffs: Vec<F::Elem>,
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
        assert!(
            sym_len.is_multiple_of(F::BYTES),
            "symbol length must be a whole number of field elements"
        );
        Self {
            cols,
            sym_len,
            sym_cols: sym_len / F::BYTES,
            rows: Vec::new(),
            rhs: Vec::new(),
            work_rows: Vec::new(),
            work_rhs: Vec::new(),
            col: Vec::new(),
            alive: Vec::new(),
            pivots: Vec::new(),
            inactive: Vec::new(),
            active_weight: Vec::new(),
            edge_rows: Vec::new(),
            edges: Vec::new(),
            active_of_pivot: Vec::new(),
            residual: Vec::new(),
            basis_rows: Vec::new(),
            needed: Vec::new(),
            schedule_parent: Vec::new(),
            schedule_rank: Vec::new(),
            schedule_size: Vec::new(),
            pivot_cols: Vec::new(),
            pivot_coeffs: Vec::new(),
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
        self.active_weight.clear();
        self.active_weight.resize(m, 0);
        self.needed.clear();
        self.needed.resize(m, false);
        self.pivots.clear();
        self.inactive.clear();
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

    fn rhs_row(&self, r: usize) -> &[u8] {
        &self.work_rhs[r * self.sym_len..(r + 1) * self.sym_len]
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
                coefficients.set(row_index, col_index, self.work_rows[source_row].get(column));
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
        // weight-one pivot on the next iteration.
        loop {
            for r in 0..m {
                if self.alive[r] {
                    self.active_weight[r] = self.work_rows[r]
                        .cols
                        .iter()
                        .filter(|&&c| self.col[c as usize] == Col::Active)
                        .count();
                }
            }
            let min_weight = (0..m)
                .filter(|&r| self.alive[r] && self.active_weight[r] >= 1)
                .map(|r| self.active_weight[r])
                .min();
            let Some(min_weight) = min_weight else {
                break;
            };

            let pivot_row = if min_weight == 2 {
                self.edge_rows.clear();
                self.edges.clear();
                for r in 0..m {
                    if self.alive[r] && self.active_weight[r] == 2 {
                        let mut active = self.work_rows[r]
                            .cols
                            .iter()
                            .copied()
                            .filter(|&c| self.col[c as usize] == Col::Active);
                        let a = active.next().expect("weight two");
                        let b = active.next().expect("weight two");
                        self.edge_rows.push(r);
                        self.edges.push((a, b));
                    }
                }
                let edge = largest_component_edge(
                    &self.edges,
                    n,
                    &mut self.schedule_parent,
                    &mut self.schedule_rank,
                    &mut self.schedule_size,
                )
                .expect("a weight-two row exists");
                self.edge_rows[edge]
            } else {
                (0..m)
                    .find(|&r| self.alive[r] && self.active_weight[r] == min_weight)
                    .expect("minimum row exists")
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
                for &column in &self.active_of_pivot[1..] {
                    self.col[column as usize] = Col::Inactive;
                    self.inactive.push(column);
                }
                continue;
            }

            let pivot_col = self.active_of_pivot[0];
            self.col[pivot_col as usize] = Col::Pivoted;
            self.alive[pivot_row] = false;
            self.pivots.push((pivot_row, pivot_col));

            self.pivot_cols.clear();
            self.pivot_coeffs.clear();
            self.pivot_cols
                .extend_from_slice(&self.work_rows[pivot_row].cols);
            self.pivot_coeffs
                .extend_from_slice(&self.work_rows[pivot_row].coeffs);
            let pivot_binary = self.work_rows[pivot_row].binary;
            let pivot_inv = self.work_rows[pivot_row].get(pivot_col).inv();

            for r in 0..m {
                if !self.alive[r] {
                    continue;
                }
                let entry = self.work_rows[r].get(pivot_col);
                if entry.is_zero() {
                    continue;
                }
                let factor = entry.mul(pivot_inv);
                if self.work_rows[r].axpy_coeffs_slices(
                    factor,
                    &self.pivot_cols,
                    &self.pivot_coeffs,
                    pivot_binary,
                ) {
                    stats.widenings += 1;
                }
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
        stats.inactivations = self.inactive.len();
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
                    let coefficient = self.work_rows[row].get(column);
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
                    let coefficient = self.work_rows[row].get(column);
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

        for &(row, pivot_col) in self.pivots.iter().rev() {
            values
                .row_mut(pivot_col as usize)
                .copy_from_slice(self.rhs_row(row));
            for (&column, &coefficient) in self.work_rows[row]
                .cols
                .iter()
                .zip(&self.work_rows[row].coeffs)
            {
                if column == pivot_col {
                    continue;
                }
                let (dst, src) = values.two_live_rows(pivot_col as usize, column as usize);
                crate::row_ops::mul_add::<F>(dst, coefficient, src);
            }
            let pivot = self.work_rows[row].get(pivot_col);
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
                    for (&column, &coefficient) in self.work_rows[row]
                        .cols
                        .iter()
                        .zip(&self.work_rows[row].coeffs)
                    {
                        if column == pivot_col {
                            continue;
                        }
                        let (dst, src) =
                            dependencies.two_live_rows(pivot_col as usize, column as usize);
                        crate::row_ops::mul_add::<F>(dst, coefficient, src);
                    }
                    let pivot = self.work_rows[row].get(pivot_col);
                    if !pivot.is_one() {
                        ops::mul_assign::<F>(dependencies.row_mut(pivot_col as usize), pivot.inv());
                    }
                }
                #[cfg(debug_assertions)]
                for source in &self.rows {
                    for basis in 0..free {
                        let mut sum = F::Elem::ZERO;
                        for (&column, &coefficient) in source.cols.iter().zip(&source.coeffs) {
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
        // answer.
        for (row, source) in self.rows.iter().enumerate() {
            if self.needed[row] {
                continue;
            }
            self.verify_rhs.fill(0);
            for (&column, &coefficient) in source.cols.iter().zip(&source.coeffs) {
                crate::row_ops::mul_add::<F>(
                    &mut self.verify_rhs,
                    coefficient,
                    values.row(column as usize),
                );
            }
            let expected = &self.rhs[row * self.sym_len..(row + 1) * self.sym_len];
            if self.verify_rhs != expected {
                return Err(SolveError::Inconsistent { row });
            }
        }

        for (work, source) in self.work_rows.iter_mut().zip(&self.rows) {
            if work.cols.capacity() < source.cols.len() {
                work.cols
                    .reserve(source.cols.len().saturating_sub(work.cols.len()));
            }
            if work.coeffs.capacity() < source.coeffs.len() {
                work.coeffs
                    .reserve(source.coeffs.len().saturating_sub(work.coeffs.len()));
            }
        }

        stats.rank = self.pivots.len() + dense_rank;
        Ok(stats)
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
