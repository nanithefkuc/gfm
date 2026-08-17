//! The sparse coefficient rows the sparse phase pivots over.
//!
//! A row is a sorted column-index list with parallel coefficients. Rows start
//! *binary* — every coefficient one, the GF(2) common case — and widen to
//! field-valued only when a field row operation first touches them (the lazy
//! GF(2)→GF(2^m) widening). Row combination merges two sorted supports through
//! caller-owned scratch, so a reused store allocates nothing after its buffers
//! reach steady size.
//!
//! The right-hand-side payload lives outside the row (a flat per-row byte
//! buffer in the driver), because the deferred log replays payload operations
//! between two rows at once and that needs disjoint mutable borrows the driver
//! owns.
//!
//! This is deliberately narrow: cheap weight queries and support edits, sized
//! for one solve. A general sparse matrix belongs in a separate crate.

use alloc::vec;
use alloc::vec::Vec;

use fgf::FieldKernels;
use fgf::field::Elem;

/// One sparse row: `coeffs[t]` sits at column `cols[t]`, `cols` strictly
/// ascending.
pub(crate) struct Row<F: FieldKernels> {
    pub cols: Vec<u32>,
    pub coeffs: Vec<F::Elem>,
    spare_cols: Vec<u32>,
    spare_coeffs: Vec<F::Elem>,
    /// Whether every coefficient is one and no field operation has touched the
    /// row — the lazy-widening bit.
    pub binary: bool,
}

impl<F: FieldKernels> Row<F> {
    /// A binary row: unit coefficients on `support` (assumed sorted, distinct).
    pub(crate) fn binary(support: Vec<u32>) -> Self {
        let coeffs = vec![F::Elem::ONE; support.len()];
        Self {
            cols: support,
            coeffs,
            spare_cols: Vec::new(),
            spare_coeffs: Vec::new(),
            binary: true,
        }
    }

    /// A field row with explicit coefficients (parallel to `support`).
    pub(crate) fn field(support: Vec<u32>, coeffs: Vec<F::Elem>) -> Self {
        debug_assert_eq!(support.len(), coeffs.len());
        Self {
            cols: support,
            coeffs,
            spare_cols: Vec::new(),
            spare_coeffs: Vec::new(),
            binary: false,
        }
    }
    pub(crate) fn empty() -> Self {
        Self {
            cols: Vec::new(),
            coeffs: Vec::new(),
            spare_cols: Vec::new(),
            spare_coeffs: Vec::new(),
            binary: true,
        }
    }

    pub(crate) fn reset_from(&mut self, source: &Self) {
        self.cols.clear();
        self.cols.extend_from_slice(&source.cols);
        self.coeffs.clear();
        self.coeffs.extend_from_slice(&source.coeffs);
        self.binary = source.binary;
    }

    /// The number of stored (nonzero) entries.
    pub(crate) fn weight(&self) -> usize {
        self.cols.len()
    }

    /// The coefficient at `col`, or zero if absent.
    pub(crate) fn get(&self, col: u32) -> F::Elem {
        match self.cols.binary_search(&col) {
            Ok(t) => self.coeffs[t],
            Err(_) => F::Elem::ZERO,
        }
    }

    /// `self += factor · src`. Returns `true` if this widened `self` from
    /// binary to field-valued.
    ///
    /// Zero results are dropped, keeping the support minimal. Each row owns
    /// both sides of the merge ping-pong, so a warmed row never steals the
    /// shared scratch capacity another row needs.
    ///
    /// `self += factor · src`, reporting columns newly added to the
    /// support through `added`; the caller uses them to extend its
    /// column-to-row index. (The no-op closure keeps the common path
    /// allocation-free without a second merge implementation.)
    pub(crate) fn axpy_coeffs_slices(
        &mut self,
        factor: F::Elem,
        src_cols: &[u32],
        src_coeffs: &[F::Elem],
        src_binary: bool,
        mut added: impl FnMut(u32),
    ) -> bool {
        self.spare_cols.clear();
        self.spare_coeffs.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.cols.len() || j < src_cols.len() {
            let take_self =
                j >= src_cols.len() || (i < self.cols.len() && self.cols[i] <= src_cols[j]);
            let take_src =
                i >= self.cols.len() || (j < src_cols.len() && src_cols[j] <= self.cols[i]);
            if take_self && take_src {
                let value = self.coeffs[i].add(factor.mul(src_coeffs[j]));
                if !value.is_zero() {
                    self.spare_cols.push(self.cols[i]);
                    self.spare_coeffs.push(value);
                }
                i += 1;
                j += 1;
            } else if take_self {
                self.spare_cols.push(self.cols[i]);
                self.spare_coeffs.push(self.coeffs[i]);
                i += 1;
            } else {
                let value = factor.mul(src_coeffs[j]);
                if !value.is_zero() {
                    added(src_cols[j]);
                    self.spare_cols.push(src_cols[j]);
                    self.spare_coeffs.push(value);
                }
                j += 1;
            }
        }
        core::mem::swap(&mut self.cols, &mut self.spare_cols);
        core::mem::swap(&mut self.coeffs, &mut self.spare_coeffs);
        if self.spare_cols.capacity() < self.cols.len() {
            self.spare_cols
                .reserve(self.cols.len().saturating_sub(self.spare_cols.len()));
        }
        if self.spare_coeffs.capacity() < self.coeffs.len() {
            self.spare_coeffs
                .reserve(self.coeffs.len().saturating_sub(self.spare_coeffs.len()));
        }
        let widened = self.binary && (!factor.is_one() || !src_binary);
        if widened {
            self.binary = false;
        }
        widened
    }
}

impl<F: FieldKernels> core::fmt::Debug for Row<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Row")
            .field("weight", &self.weight())
            .field("binary", &self.binary)
            .finish_non_exhaustive()
    }
}
