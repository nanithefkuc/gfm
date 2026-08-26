//! The sparse coefficient rows the sparse phase pivots over.
//!
//! A binary row keeps two parts: `cols`, the sorted columns still *active*
//! in the schedule, and `frozen`, bit-packed words over the columns that
//! have already been inactivated, indexed by the solver's global frozen
//! ordinal. Inactivation moves a column's entry out of every packed row's
//! list and into its bit words, so later combinations stop walking the
//! accumulated fill one index at a time and XOR whole words instead — the
//! same split the reference PI solvers draw between sparse lists and a
//! dense trailing block. A field operation widens the row once into a
//! flat list (`cols` over *all* columns with parallel `coeffs`), which
//! never packs again: field coefficients cannot ride a bit.
//!
//! Row combination merges two sorted supports through caller-owned scratch,
//! so a reused store allocates nothing after its buffers reach steady size.
//!
//! The right-hand-side payload lives outside the row (a flat per-row byte
//! buffer in the driver), because the deferred log replays payload operations
//! between two rows at once and that needs disjoint mutable borrows the driver
//! owns.
//!
//! This is deliberately narrow: cheap weight queries and support edits, sized
//! for one solve. A general sparse matrix belongs in a separate crate.

use alloc::vec::Vec;

use fgf::FieldKernels;
use fgf::field::Elem;

/// Sentinel frozen ordinal for a column that is not (yet) inactivated.
pub(crate) const NOT_FROZEN: u32 = u32::MAX;

/// One sparse row over GF(2)-majority systems: unit coefficients on `cols`
/// until a field operation touches the row.
///
/// A packed row (`binary == true`) carries implicit unit coefficients:
/// `coeffs` is empty, `cols` holds exactly the active columns, and `frozen`
/// holds one bit per inactivated column keyed by the global frozen ordinal
/// (word `w` covers ordinals `[64w, 64w + 64)`, padding zero). A widened row
/// (`binary == false`) carries explicit coefficients parallel to `cols`,
/// which then lists every column — active and frozen alike.
pub(crate) struct Row<F: FieldKernels> {
    pub cols: Vec<u32>,
    pub coeffs: Vec<F::Elem>,
    /// Bit-packed inactivated-column support of a packed row; unused
    /// (empty) once the row widens.
    pub frozen: Vec<u64>,
    spare_cols: Vec<u32>,
    spare_coeffs: Vec<F::Elem>,
    /// Whether every coefficient is one and the row still splits into the
    /// active list plus frozen words — the lazy-widening bit.
    pub binary: bool,
}

impl<F: FieldKernels> Row<F> {
    /// A binary row: unit coefficients on `support` (assumed sorted, distinct,
    /// and all active — packing splits it later, in `prepare_work`).
    pub(crate) fn binary(support: Vec<u32>) -> Self {
        Self {
            cols: support,
            coeffs: Vec::new(),
            frozen: Vec::new(),
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
            frozen: Vec::new(),
            spare_cols: Vec::new(),
            spare_coeffs: Vec::new(),
            binary: false,
        }
    }
    pub(crate) fn empty() -> Self {
        Self {
            cols: Vec::new(),
            coeffs: Vec::new(),
            frozen: Vec::new(),
            spare_cols: Vec::new(),
            spare_coeffs: Vec::new(),
            binary: true,
        }
    }

    /// Copies an *input* row (flat list over all columns, never packed).
    pub(crate) fn reset_from(&mut self, source: &Self) {
        self.cols.clear();
        self.cols.extend_from_slice(&source.cols);
        self.coeffs.clear();
        self.coeffs.extend_from_slice(&source.coeffs);
        self.frozen.clear();
        self.binary = source.binary;
    }

    /// The number of stored (nonzero) entries.
    pub(crate) fn weight(&self) -> usize {
        self.cols.len()
    }

    /// The coefficient at `col`, given the column's global frozen ordinal.
    ///
    /// Callers pass [`NOT_FROZEN`] for columns known active. A packed row
    /// answers a frozen column from its bits; everything else searches the
    /// list. Input rows must go through [`Self::get_input`]: a valid
    /// ordinal describes the *working* system, not the untouched input.
    pub(crate) fn get(&self, col: u32, frozen_ordinal: u32) -> F::Elem {
        if self.binary && frozen_ordinal != NOT_FROZEN {
            return self.frozen_bit(frozen_ordinal);
        }
        match self.cols.binary_search(&col) {
            Ok(t) => {
                if self.binary {
                    F::Elem::ONE
                } else {
                    self.coeffs[t]
                }
            }
            Err(_) => F::Elem::ZERO,
        }
    }

    /// The coefficient at `col` of an input row: pure list lookup.
    pub(crate) fn get_input(&self, col: u32) -> F::Elem {
        match self.cols.binary_search(&col) {
            Ok(t) => {
                if self.binary {
                    F::Elem::ONE
                } else {
                    self.coeffs[t]
                }
            }
            Err(_) => F::Elem::ZERO,
        }
    }

    /// Whether the packed row's list still contains `col` — membership in
    /// the *active* part only, ignoring frozen bits. Used by the freeze
    /// migration before the entries move.
    pub(crate) fn contains_in_list(&self, col: u32) -> bool {
        self.cols.binary_search(&col).is_ok()
    }

    fn frozen_bit(&self, ordinal: u32) -> F::Elem {
        let word = ordinal as usize / 64;
        let bit = ordinal % 64;
        if word < self.frozen.len() && self.frozen[word] >> bit & 1 == 1 {
            F::Elem::ONE
        } else {
            F::Elem::ZERO
        }
    }

    /// Moves every listed column out of the active list and into the frozen
    /// bits. The columns must currently sit in the list; the row stays packed.
    /// Returns how many entries moved (the row's active-weight loss).
    pub(crate) fn freeze_from_list(&mut self, columns: &[u32], ordinals: &[u32]) -> usize {
        debug_assert!(self.binary);
        self.spare_cols.clear();
        let mut moved = 0usize;
        // Direct field access: the scan reads `cols` while the migration
        // writes `frozen`, which are disjoint.
        for index in 0..self.cols.len() {
            let col = self.cols[index];
            if columns.binary_search(&col).is_ok() {
                let ordinal = ordinals[col as usize];
                let word = ordinal as usize / 64;
                if word >= self.frozen.len() {
                    self.frozen.resize(word + 1, 0);
                }
                self.frozen[word] |= 1 << (ordinal % 64);
                moved += 1;
            } else {
                self.spare_cols.push(col);
            }
        }
        core::mem::swap(&mut self.cols, &mut self.spare_cols);
        if self.spare_cols.capacity() < self.cols.len() {
            self.spare_cols
                .reserve(self.cols.len().saturating_sub(self.spare_cols.len()));
        }
        moved
    }

    /// Widens a packed row into the flat list form: all columns (active plus
    /// frozen, re-sorted) with unit coefficients. Never reverses.
    pub(crate) fn materialize(&mut self, frozen_cols: &[u32]) -> bool {
        if !self.binary {
            return false;
        }
        debug_assert!(self.coeffs.is_empty());
        self.spare_cols.clear();
        for (word, &bits) in self.frozen.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                self.spare_cols.push(frozen_cols[word * 64 + bit as usize]);
            }
        }
        self.cols.append(&mut self.spare_cols);
        self.cols.sort_unstable();
        self.coeffs.resize(self.cols.len(), F::Elem::ONE);
        self.frozen.clear();
        self.binary = false;
        true
    }

    /// Runs `f` over every nonzero entry `(column, coefficient)` of the row —
    /// the active list, then the frozen bits of a packed row, then (for a
    /// widened row) nothing further. Order is not column-sorted across the
    /// two parts; every consumer here folds commutatively.
    pub(crate) fn for_each_entry(&self, frozen_cols: &[u32], mut f: impl FnMut(u32, F::Elem)) {
        if self.binary {
            for &col in &self.cols {
                f(col, F::Elem::ONE);
            }
            for (word, &bits) in self.frozen.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    let bit = bits.trailing_zeros();
                    bits &= bits - 1;
                    f(frozen_cols[word * 64 + bit as usize], F::Elem::ONE);
                }
            }
        } else {
            for (&col, &value) in self.cols.iter().zip(&self.coeffs) {
                f(col, value);
            }
        }
    }

    /// The packed-row GF(2) combination: `self += src` with unit coefficients
    /// on both sides. The active lists merge as a support XOR; the frozen
    /// words XOR wholesale, cancellation included. Returns the new active
    /// weight. Columns newly added to the active list go through `added`.
    pub(crate) fn axpy_xor_parts(
        &mut self,
        src_cols: &[u32],
        src_frozen: &[u64],
        mut added: impl FnMut(u32),
        mut is_active: impl FnMut(u32) -> bool,
    ) -> usize {
        debug_assert!(self.binary);
        let mut active = 0usize;
        self.spare_cols.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.cols.len() || j < src_cols.len() {
            let take_self =
                j >= src_cols.len() || (i < self.cols.len() && self.cols[i] <= src_cols[j]);
            let take_src =
                i >= self.cols.len() || (j < src_cols.len() && src_cols[j] <= self.cols[i]);
            if take_self && take_src {
                // 1 + 1 = 0: the entry cancels and drops out.
                i += 1;
                j += 1;
            } else if take_self {
                if is_active(self.cols[i]) {
                    active += 1;
                }
                self.spare_cols.push(self.cols[i]);
                i += 1;
            } else {
                if is_active(src_cols[j]) {
                    active += 1;
                }
                added(src_cols[j]);
                self.spare_cols.push(src_cols[j]);
                j += 1;
            }
        }
        core::mem::swap(&mut self.cols, &mut self.spare_cols);
        if self.spare_cols.capacity() < self.cols.len() {
            self.spare_cols
                .reserve(self.cols.len().saturating_sub(self.spare_cols.len()));
        }
        // Frozen half: bulk XOR over the shared ordinal space, then trim
        // words that fully cancelled.
        if self.frozen.len() < src_frozen.len() {
            self.frozen.resize(src_frozen.len(), 0);
        }
        for (dst, &src) in self.frozen.iter_mut().zip(src_frozen) {
            *dst ^= src;
        }
        while self.frozen.last() == Some(&0) {
            self.frozen.pop();
        }
        active
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn axpy_coeffs_slices(
        &mut self,
        factor: F::Elem,
        src_cols: &[u32],
        src_coeffs: &[F::Elem],
        src_binary: bool,
        mut added: impl FnMut(u32),
        mut is_active: impl FnMut(u32) -> bool,
    ) -> (bool, usize) {
        debug_assert!(
            !self.binary,
            "packed rows take axpy_xor_parts or widen first"
        );
        let mut active = 0usize;
        self.spare_cols.clear();
        self.spare_coeffs.clear();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.cols.len() || j < src_cols.len() {
            let take_self =
                j >= src_cols.len() || (i < self.cols.len() && self.cols[i] <= src_cols[j]);
            let take_src =
                i >= self.cols.len() || (j < src_cols.len() && src_cols[j] <= self.cols[i]);
            if take_self && take_src {
                let term = if src_binary {
                    factor
                } else {
                    factor.mul(src_coeffs[j])
                };
                let value = self.coeffs[i].add(term);
                if !value.is_zero() {
                    if is_active(self.cols[i]) {
                        active += 1;
                    }
                    self.spare_cols.push(self.cols[i]);
                    self.spare_coeffs.push(value);
                }
                i += 1;
                j += 1;
            } else if take_self {
                if is_active(self.cols[i]) {
                    active += 1;
                }
                self.spare_cols.push(self.cols[i]);
                self.spare_coeffs.push(self.coeffs[i]);
                i += 1;
            } else {
                let value = if src_binary {
                    factor
                } else {
                    factor.mul(src_coeffs[j])
                };
                if !value.is_zero() {
                    if is_active(src_cols[j]) {
                        active += 1;
                    }
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
                .reserve(self.coeffs.len().saturating_sub(self.coeffs.len()));
        }
        (false, active)
    }

    /// Reserves working-buffer capacity against an input row's shape.
    pub(crate) fn reserve_like(&mut self, source: &Self) {
        if self.cols.capacity() < source.cols.len() {
            self.cols
                .reserve(source.cols.len().saturating_sub(self.cols.len()));
        }
        if self.coeffs.capacity() < source.coeffs.len() {
            self.coeffs
                .reserve(source.coeffs.len().saturating_sub(self.coeffs.len()));
        }
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
