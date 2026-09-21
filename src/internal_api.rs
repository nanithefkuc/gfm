//! Unstable extension traits re-exported by the `internals` facade.

use fgf::FieldKernels;

use crate::SolveError;
use crate::bits::{BitMatrix, Ple as BitPle, PleScratch as BitPleScratch};
use crate::dense::{Matrix, Perm, Ple, PleScratch};
use crate::hybrid::{Hybrid, Solution, SolveStats};

/// Physical-layout inspection of a GF(2^m) matrix.
pub trait MatrixInternals {
    /// Address of the first byte of the physical backing region. A multiple
    /// of [`crate::dense::layout::ALIGN`] by construction.
    fn base_addr(&self) -> usize;

    /// The physical row logical row `row` maps to.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds.
    fn physical_row_index(&self, row: usize) -> usize;

    /// The whole physical backing region, padding included, laid out as
    /// physical rows of the pitch.
    fn pitched_buffer(&self) -> &[u8];
}

impl<F: FieldKernels> MatrixInternals for Matrix<F> {
    fn base_addr(&self) -> usize {
        Matrix::base_addr(self)
    }

    fn physical_row_index(&self, row: usize) -> usize {
        Matrix::physical_row_index(self, row)
    }

    fn pitched_buffer(&self) -> &[u8] {
        Matrix::pitched_buffer(self)
    }
}

/// Physical-layout inspection of a bit-packed GF(2) matrix.
pub trait BitMatrixInternals {
    /// Address of the first byte of the physical backing region. A multiple
    /// of [`crate::bits::ALIGN`] by construction.
    fn base_addr(&self) -> usize;

    /// The physical row logical row `row` maps to.
    ///
    /// # Panics
    ///
    /// Panics if `row` is out of bounds.
    fn physical_row_index(&self, row: usize) -> usize;

    /// The whole physical backing region, padding included, laid out as
    /// physical rows.
    fn pitched_buffer(&self) -> &[u8];
}

impl BitMatrixInternals for BitMatrix {
    fn base_addr(&self) -> usize {
        BitMatrix::base_addr(self)
    }

    fn physical_row_index(&self, row: usize) -> usize {
        BitMatrix::physical_row_index(self, row)
    }

    fn pitched_buffer(&self) -> &[u8] {
        BitMatrix::pitched_buffer(self)
    }
}

/// Decomposition storage and the trailing-update candidates of the GF(2^m)
/// elimination.
pub trait PleInternals<F: FieldKernels> {
    /// The in-place `L`/`U` storage: factors below the diagonal, `U` on and
    /// above it, in the eliminated row and column order.
    fn lu(&self) -> &Matrix<F>;

    /// The row permutation, as a LAPACK-style swap list.
    fn p(&self) -> &Perm;

    /// The column permutation, as a LAPACK-style swap list.
    fn q(&self) -> &Perm;

    /// Decomposes with an explicit panel width. The result is independent of
    /// the width, byte for byte.
    fn decompose_with_panel_width(
        a: Matrix<F>,
        scratch: &mut PleScratch<F>,
        panel_width: usize,
    ) -> Self;

    /// Decomposes with the Newton–John trailing-update candidate instead of
    /// the dispatched one.
    ///
    /// # Panics
    ///
    /// Panics unless field elements occupy one byte.
    fn decompose_newton_john(a: Matrix<F>, scratch: &mut PleScratch<F>) -> Self;
}

impl<F: FieldKernels> PleInternals<F> for Ple<F> {
    fn lu(&self) -> &Matrix<F> {
        self.lu_matrix()
    }

    fn p(&self) -> &Perm {
        self.p_perm()
    }

    fn q(&self) -> &Perm {
        self.q_perm()
    }

    fn decompose_with_panel_width(
        a: Matrix<F>,
        scratch: &mut PleScratch<F>,
        panel_width: usize,
    ) -> Self {
        Ple::decompose_with_panel_width(a, scratch, panel_width)
    }

    fn decompose_newton_john(a: Matrix<F>, scratch: &mut PleScratch<F>) -> Self {
        Ple::decompose_newton_john(a, scratch)
    }
}

/// Decomposition storage and the trailing-update candidates of the GF(2)
/// elimination.
pub trait BitPleInternals {
    /// The in-place `L`/`U` storage: factors below the diagonal, `U` on and
    /// above it, in the eliminated row and column order.
    fn lu(&self) -> &BitMatrix;

    /// The row permutation, as a LAPACK-style swap list.
    fn p(&self) -> &Perm;

    /// The column permutation, as a LAPACK-style swap list.
    fn q(&self) -> &Perm;

    /// Decomposes with an explicit panel width. The result is independent of
    /// the width, byte for byte.
    fn decompose_with_panel_width(
        a: BitMatrix,
        scratch: &mut BitPleScratch,
        panel_width: usize,
    ) -> Self;

    /// Decomposes with the untabled trailing update instead of the
    /// dispatched one.
    fn decompose_plain(a: BitMatrix, scratch: &mut BitPleScratch) -> Self;

    /// Decomposes with the combination-table trailing update instead of the
    /// dispatched one.
    fn decompose_m4ri(a: BitMatrix, scratch: &mut BitPleScratch) -> Self;
}

impl BitPleInternals for BitPle {
    fn lu(&self) -> &BitMatrix {
        self.lu_matrix()
    }

    fn p(&self) -> &Perm {
        self.p_perm()
    }

    fn q(&self) -> &Perm {
        self.q_perm()
    }

    fn decompose_with_panel_width(
        a: BitMatrix,
        scratch: &mut BitPleScratch,
        panel_width: usize,
    ) -> Self {
        BitPle::decompose_with_panel_width(a, scratch, panel_width)
    }

    fn decompose_plain(a: BitMatrix, scratch: &mut BitPleScratch) -> Self {
        BitPle::decompose_plain(a, scratch)
    }

    fn decompose_m4ri(a: BitMatrix, scratch: &mut BitPleScratch) -> Self {
        BitPle::decompose_m4ri(a, scratch)
    }
}

/// Schedule and row-operation counters of the hybrid solver.
pub trait HybridInternals<F: FieldKernels> {
    /// Solves and reports the schedule and row-operation counters. `defer`
    /// selects the deferred payload path.
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics if the requested output geometry overflows `usize`.
    fn solve_reported(&mut self, defer: bool) -> Result<(Solution<F>, SolveStats), SolveError>;

    /// Allocation-free form of [`Self::solve_reported`].
    ///
    /// # Errors
    ///
    /// [`SolveError::Inconsistent`] if the system has no solution.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Hybrid::solve_into`].
    fn solve_into_reported(
        &mut self,
        defer: bool,
        values: &mut Matrix<F>,
        determined: &mut [bool],
    ) -> Result<SolveStats, SolveError>;
}

impl<F: FieldKernels> HybridInternals<F> for Hybrid<F> {
    fn solve_reported(&mut self, defer: bool) -> Result<(Solution<F>, SolveStats), SolveError> {
        Hybrid::solve_reported(self, defer)
    }

    fn solve_into_reported(
        &mut self,
        defer: bool,
        values: &mut Matrix<F>,
        determined: &mut [bool],
    ) -> Result<SolveStats, SolveError> {
        Hybrid::solve_into_reported(self, defer, values, determined)
    }
}

/// The row-update dispatcher elimination and scaling drive.
pub mod row_ops {
    use fgf::FieldKernels;

    /// Accumulates `factor * src` into `dst` through the dispatcher the
    /// elimination uses: the Rayon boundary with the `parallel` feature,
    /// `fgf`'s kernel directly without it.
    pub fn mul_add<F: FieldKernels>(dst: &mut [u8], factor: F::Elem, src: &[u8]) {
        crate::row_ops::mul_add::<F>(dst, factor, src);
    }
}
