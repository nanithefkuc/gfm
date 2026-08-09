//! [`Echelon`] — the streaming Gaussian-elimination accumulator.
//!
//! One equation at a time: reduce the incoming row against retained pivots in
//! pivot-column order until the first free nonzero column; if one survives,
//! the row is innovative and becomes a new pivot, otherwise it is dependent
//! (or inconsistent, if its right-hand side did not also vanish). This is the
//! whole of on-the-fly decoding — the rank reaches full the instant the last
//! innovative packet arrives, with no batch phase.
//!
//! The `reduced` flag collapses `ccrlnc`'s two bases into one type: a decoder
//! sets it to substitute every recovered unit row and propagate new unit rows
//! backward, so a variable's value is readable as soon as it is determined; a
//! recoder clears it and keeps forward echelon only.
//!
//! `absorb` reuses two internal scratch rows, so a steady-state stream
//! allocates nothing.

use alloc::vec;
use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::{FieldKernels, ops};

use crate::GeometryError;
use crate::dense::Matrix;
use crate::incremental::Innovation;

const STACK_GATHER: usize = 64;
#[inline]
fn mul_add_coefficients<F: FieldKernels>(dst: &mut [u8], factor: F::Elem, src: &[u8]) {
    if dst.len() / F::BYTES <= 64 {
        for (dst, src) in dst
            .chunks_exact_mut(F::BYTES)
            .zip(src.chunks_exact(F::BYTES))
        {
            F::write(dst, F::read(dst).add(factor.mul(F::read(src))));
        }
    } else {
        crate::row_ops::mul_add::<F>(dst, factor, src);
    }
}

#[inline]
fn mul_add_cyclic<F: FieldKernels>(
    dst: &mut [u8],
    factor: F::Elem,
    src: &[u8],
    start: usize,
    len: usize,
) {
    let start = start * F::BYTES;
    let len = len * F::BYTES;
    let first = len.min(dst.len() - start);
    mul_add_coefficients::<F>(
        &mut dst[start..start + first],
        factor,
        &src[start..start + first],
    );
    if first != len {
        mul_add_coefficients::<F>(&mut dst[..len - first], factor, &src[..len - first]);
    }
}

/// A streaming echelon accumulator over `cols` coefficient columns, carrying a
/// right-hand-side payload of `rhs_cols` field elements per row.
pub struct Echelon<F: FieldKernels> {
    /// One retained pivot row per rank, coefficients in `cols` columns.
    pivots: Matrix<F>,
    /// The matching right-hand-side payload for each pivot row.
    rhs: Matrix<F>,
    /// Physical ring column → pivot row slot.
    pivot_of_col: Vec<Option<usize>>,
    free_slots: Vec<usize>,
    next_slot: usize,
    unit_physical: Vec<Option<usize>>,
    span_of_slot: Vec<usize>,
    rank: usize,
    cols: usize,
    rhs_cols: usize,
    column_offset: usize,
    reduced: bool,
    scratch_coeffs: Vec<u8>,
    scratch_rhs: Vec<u8>,
    recovered: Vec<bool>,
    newly_recovered: Vec<usize>,
}

/// A retained echelon row in logical column order.
pub struct RetainedRow<'a, F: FieldKernels> {
    pivot: usize,
    span: usize,
    coefficients: &'a [u8],
    rhs: &'a [u8],
    cols: usize,
    physical: usize,
    field: core::marker::PhantomData<F>,
}

impl<'a, F: FieldKernels> RetainedRow<'a, F> {
    /// The row's logical pivot column.
    #[must_use]
    pub const fn pivot(&self) -> usize {
        self.pivot
    }

    /// The packed right-hand-side row.
    #[must_use]
    pub const fn rhs(&self) -> &'a [u8] {
        self.rhs
    }

    /// Packed coefficients from the pivot through the last nonzero entry, in
    /// logical order. Concatenating the two slices yields one field element
    /// sequence; the second is empty unless the ring storage wraps.
    #[must_use]
    pub fn coefficient_slices(&self) -> (&'a [u8], &'a [u8]) {
        let start = self.physical * F::BYTES;
        let len = self.span * F::BYTES;
        let first = len.min(self.coefficients.len() - start);
        (
            &self.coefficients[start..start + first],
            &self.coefficients[..len - first],
        )
    }

    /// Adds `factor * coefficients` to a packed logical-order output row.
    ///
    /// # Panics
    ///
    /// Panics unless `out` contains exactly `cols` field elements.
    pub fn mul_add_coefficients_into(&self, out: &mut [u8], factor: F::Elem) {
        assert_eq!(out.len(), self.cols * F::BYTES, "coefficient output length");
        let dst_start = self.pivot * F::BYTES;
        let src_start = self.physical;
        let len = self.span * F::BYTES;
        let first = len.min((self.cols - src_start) * F::BYTES);
        crate::row_ops::mul_add::<F>(
            &mut out[dst_start..dst_start + first],
            factor,
            &self.coefficients[src_start * F::BYTES..src_start * F::BYTES + first],
        );
        if first != len {
            crate::row_ops::mul_add::<F>(
                &mut out[dst_start + first..dst_start + len],
                factor,
                &self.coefficients[..len - first],
            );
        }
    }
}

impl<F: FieldKernels> core::fmt::Debug for RetainedRow<'_, F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RetainedRow")
            .field("pivot", &self.pivot)
            .finish_non_exhaustive()
    }
}

impl<F: FieldKernels> Echelon<F> {
    /// A fresh accumulator over `cols` coefficient columns and a right-hand
    /// side of `rhs_len` bytes per row.
    ///
    /// `reduced` selects recovered-unit substitution and propagation
    /// (decoder) versus forward echelon only (recoder).
    ///
    /// # Errors
    ///
    /// [`GeometryError::Ragged`] if `rhs_len` is not a whole number of
    /// elements, [`GeometryError::Overflow`] if the state geometry overflows.
    pub fn new(cols: usize, rhs_len: usize, reduced: bool) -> Result<Self, GeometryError> {
        if !rhs_len.is_multiple_of(F::BYTES) {
            return Err(GeometryError::Ragged {
                len: rhs_len,
                element_bytes: F::BYTES,
            });
        }
        let rhs_cols = rhs_len / F::BYTES;
        let pivots = Matrix::<F>::zeros(cols, cols)?;
        let rhs = Matrix::<F>::zeros(cols, rhs_cols)?;
        Ok(Self {
            pivots,
            rhs,
            pivot_of_col: vec![None; cols],
            free_slots: Vec::with_capacity(cols),
            next_slot: 0,
            unit_physical: vec![None; cols],
            span_of_slot: vec![0; cols],
            rank: 0,
            cols,
            rhs_cols,
            column_offset: 0,
            reduced,
            scratch_coeffs: vec![0u8; cols * F::BYTES],
            scratch_rhs: vec![0u8; rhs_cols * F::BYTES],
            recovered: vec![false; cols],
            newly_recovered: Vec::with_capacity(cols),
        })
    }

    /// The current rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Whether the system is fully determined (`rank == cols`).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.rank == self.cols
    }

    /// The pivot columns in ascending logical order.
    pub fn pivot_columns(&self) -> impl Iterator<Item = usize> + '_ {
        (self.column_offset..self.cols)
            .chain(0..self.column_offset)
            .enumerate()
            .filter_map(|(column, physical)| {
                self.pivot_of_col[physical].is_some().then_some(column)
            })
    }

    /// Whether `column` already owns a retained pivot.
    #[must_use]
    pub fn has_pivot(&self, column: usize) -> bool {
        self.pivot_of_col[self.logical_to_physical(column)].is_some()
    }

    /// Retained rows in ascending logical pivot-column order.
    pub fn retained_rows(&self) -> impl Iterator<Item = RetainedRow<'_, F>> + '_ {
        (self.column_offset..self.cols)
            .chain(0..self.column_offset)
            .enumerate()
            .filter_map(move |(column, physical)| {
                self.pivot_of_col[physical].map(|slot| RetainedRow {
                    pivot: column,
                    span: self.span_of_slot[slot],
                    coefficients: self.pivots.row(slot),
                    rhs: self.rhs.row(slot),
                    cols: self.cols,
                    physical,
                    field: core::marker::PhantomData,
                })
            })
    }

    /// Discards the first `count` variables and shifts every surviving
    /// variable left by `count` columns.
    ///
    /// New columns opened at the right are zero in every retained equation.
    /// Rows pivoted in the discarded prefix are removed. Every surviving row
    /// is already zero before its pivot, so dropping the prefix preserves the
    /// represented equations in both forward and reduced modes.
    ///
    /// This is the storage operation sliding-window codecs need; absolute
    /// source numbering and window validation remain caller policy.
    ///
    /// # Panics
    ///
    /// Panics if `count > cols`.
    pub fn advance_prefix(&mut self, count: usize) {
        assert!(count <= self.cols, "advanced prefix exceeds column count");
        if count == 0 {
            return;
        }
        if count == self.cols {
            self.rank = 0;
            self.free_slots.clear();
            self.next_slot = 0;
            self.unit_physical.fill(None);
            self.pivot_of_col.fill(None);
            self.recovered.fill(false);
            self.newly_recovered.clear();
            self.column_offset = 0;
            return;
        }

        for column in 0..count {
            let physical = self.logical_to_physical(column);
            if let Some(slot) = self.pivot_of_col[physical].take() {
                self.free_slots.push(slot);
                self.rank -= 1;
            }
            self.recovered[physical] = false;
        }
        self.column_offset = (self.column_offset + count) % self.cols;
        self.newly_recovered.clear();
    }
    /// Retains a unit equation at a currently free pivot column.
    ///
    /// This is the allocation-free systematic-packet path: it avoids building
    /// and reducing a dense coefficient scratch row.
    ///
    /// # Panics
    ///
    /// Panics if `column` is out of range, already pivoted, or `rhs` has the
    /// wrong length.
    pub fn absorb_unit(&mut self, column: usize, rhs: &[u8]) -> Innovation {
        assert!(column < self.cols, "pivot column out of range");
        assert!(
            self.pivot_of_col[self.logical_to_physical(column)].is_none(),
            "unit pivot column is occupied"
        );
        assert_eq!(rhs.len(), self.scratch_rhs.len(), "right-hand side length");
        self.newly_recovered.clear();
        let physical = (self.column_offset + column) % self.cols;
        let slot = self.take_slot();
        if self.unit_physical[slot] != Some(physical) {
            self.pivots.row_mut(slot).fill(0);
            self.pivots.set(slot, physical, F::Elem::ONE);
            self.unit_physical[slot] = Some(physical);
        }
        self.span_of_slot[slot] = 1;
        self.rhs.row_mut(slot).copy_from_slice(rhs);
        self.pivot_of_col[physical] = Some(slot);

        if self.reduced {
            self.propagate_unit(physical);
        }
        self.rank += 1;
        Innovation::Innovative { pivot: column }
    }

    /// Absorbs one equation: coefficients (`cols` elements) and a right-hand
    /// side (`rhs_cols` elements), both packed little-endian.
    ///
    /// # Panics
    ///
    /// Panics unless `coeffs` is `cols * F::BYTES` bytes and `rhs` is
    /// `rhs_cols * F::BYTES` bytes.
    pub fn absorb(&mut self, coeffs: &[u8], rhs: &[u8]) -> Innovation {
        assert_eq!(coeffs.len(), self.cols * F::BYTES, "coefficient length");
        assert_eq!(rhs.len(), self.scratch_rhs.len(), "right-hand side length");
        let offset = self.column_offset * F::BYTES;
        let first = self.scratch_coeffs.len() - offset;
        self.scratch_coeffs[offset..].copy_from_slice(&coeffs[..first]);
        self.scratch_coeffs[..offset].copy_from_slice(&coeffs[first..]);
        self.scratch_rhs.copy_from_slice(rhs);
        self.newly_recovered.clear();
        // Reduce coefficients in pivot order. For codec-sized systems, collect
        // the matching RHS operation into one gather so each payload byte is
        // loaded and stored once rather than once per pivot.
        let mut rhs_factors = [F::Elem::ZERO; STACK_GATHER];
        let mut rhs_sources = [&[][..]; STACK_GATHER];
        let mut rhs_count = 0;
        let mut pivot = None;
        for (c, physical) in (self.column_offset..self.cols)
            .chain(0..self.column_offset)
            .enumerate()
        {
            let start = physical * F::BYTES;
            let factor = F::read(&self.scratch_coeffs[start..start + F::BYTES]);
            if factor.is_zero() {
                continue;
            }
            let Some(slot) = self.pivot_of_col[physical] else {
                pivot = Some(c);
                break;
            };
            if self.span_of_slot[slot] == 1 {
                self.scratch_coeffs[start..start + F::BYTES].fill(0);
            } else {
                mul_add_cyclic::<F>(
                    &mut self.scratch_coeffs,
                    factor,
                    self.pivots.row(slot),
                    physical,
                    self.span_of_slot[slot],
                );
            }
            if self.cols <= STACK_GATHER {
                rhs_factors[rhs_count] = factor;
                rhs_sources[rhs_count] = self.rhs.row(slot);
                rhs_count += 1;
            } else {
                crate::row_ops::mul_add::<F>(&mut self.scratch_rhs, factor, self.rhs.row(slot));
            }
        }
        if self.reduced
            && let Some(pivot_column) = pivot
        {
            for (_, physical) in (self.column_offset..self.cols)
                .chain(0..self.column_offset)
                .enumerate()
                .skip(pivot_column + 1)
            {
                if !self.recovered[physical] {
                    continue;
                }
                let start = physical * F::BYTES;
                let factor = F::read(&self.scratch_coeffs[start..start + F::BYTES]);
                if factor.is_zero() {
                    continue;
                }
                let Some(slot) = self.pivot_of_col[physical] else {
                    continue;
                };
                self.scratch_coeffs[start..start + F::BYTES].fill(0);
                if self.cols <= STACK_GATHER {
                    rhs_factors[rhs_count] = factor;
                    rhs_sources[rhs_count] = self.rhs.row(slot);
                    rhs_count += 1;
                } else {
                    crate::row_ops::mul_add::<F>(&mut self.scratch_rhs, factor, self.rhs.row(slot));
                }
            }
        }
        if rhs_count != 0 {
            ops::mul_add_gather::<F>(
                &mut self.scratch_rhs,
                &rhs_factors[..rhs_count],
                &rhs_sources[..rhs_count],
            );
        }
        // The first surviving non-pivot coefficient is the new pivot column.
        let Some(pivot) = pivot else {
            return if self.scratch_rhs.iter().any(|&b| b != 0) {
                Innovation::Inconsistent
            } else {
                Innovation::Dependent
            };
        };

        let (slot, pivot_physical, span) = self.retain_scratch_row(pivot);

        if self.reduced && span == 1 {
            self.unit_physical[slot] = Some(pivot_physical);
            self.propagate_unit(pivot_physical);
        }
        self.rank += 1;
        Innovation::Innovative { pivot }
    }

    #[inline]
    fn take_slot(&mut self) -> usize {
        self.free_slots.pop().unwrap_or_else(|| {
            let slot = self.next_slot;
            self.next_slot += 1;
            slot
        })
    }

    #[inline]
    fn retain_scratch_row(&mut self, pivot: usize) -> (usize, usize, usize) {
        let pivot_physical = self.logical_to_physical(pivot);
        let pivot_start = pivot_physical * F::BYTES;
        let pivot_val = F::read(&self.scratch_coeffs[pivot_start..pivot_start + F::BYTES]);
        if !pivot_val.is_one() {
            let inv = pivot_val.inv();
            ops::mul_assign::<F>(&mut self.scratch_coeffs, inv);
            ops::mul_assign::<F>(&mut self.scratch_rhs, inv);
        }
        let last = (0..self.column_offset)
            .rev()
            .chain((self.column_offset..self.cols).rev())
            .enumerate()
            .find(|&(_, physical)| {
                let start = physical * F::BYTES;
                !F::read(&self.scratch_coeffs[start..start + F::BYTES]).is_zero()
            })
            .map(|(from_end, _)| self.cols - 1 - from_end)
            .expect("the pivot is nonzero");
        let span = last - pivot + 1;
        let slot = self.take_slot();
        self.pivots
            .row_mut(slot)
            .copy_from_slice(&self.scratch_coeffs);
        self.rhs.row_mut(slot).copy_from_slice(&self.scratch_rhs);
        self.unit_physical[slot] = None;
        self.span_of_slot[slot] = span;
        self.pivot_of_col[pivot_physical] = Some(slot);
        (slot, pivot_physical, span)
    }

    fn propagate_unit(&mut self, physical: usize) {
        self.recovered[physical] = true;
        self.newly_recovered
            .push(self.physical_to_logical(physical));
        let mut next = 0;
        while next < self.newly_recovered.len() {
            let known_physical = self.logical_to_physical(self.newly_recovered[next]);
            next += 1;
            let known_slot =
                self.pivot_of_col[known_physical].expect("a recovered column has a pivot");
            for other_physical in 0..self.cols {
                let Some(slot) = self.pivot_of_col[other_physical] else {
                    continue;
                };
                if slot == known_slot {
                    continue;
                }
                let factor = self.pivots.get(slot, known_physical);
                if factor.is_zero() {
                    continue;
                }
                self.pivots.set(slot, known_physical, F::Elem::ZERO);
                let pivot = self.physical_to_logical(other_physical);
                while self.span_of_slot[slot] > 1 {
                    let last = self.logical_to_physical(pivot + self.span_of_slot[slot] - 1);
                    if !self.pivots.get(slot, last).is_zero() {
                        break;
                    }
                    self.span_of_slot[slot] -= 1;
                }
                let (dst, src) = self.rhs.two_live_rows(slot, known_slot);
                crate::row_ops::mul_add::<F>(dst, factor, src);
                if !self.recovered[other_physical] && self.span_of_slot[slot] == 1 {
                    self.recovered[other_physical] = true;
                    self.unit_physical[slot] = Some(other_physical);
                    self.newly_recovered
                        .push(self.physical_to_logical(other_physical));
                }
            }
        }
    }

    #[inline]
    fn logical_to_physical(&self, column: usize) -> usize {
        (self.column_offset + column) % self.cols
    }

    #[inline]
    fn physical_to_logical(&self, column: usize) -> usize {
        (column + self.cols - self.column_offset) % self.cols
    }

    /// The variables that are fully recovered: pivot columns whose retained
    /// row has collapsed to a lone unit entry, paired with the recovered
    /// payload borrowed from that row. Meaningful only for a `reduced`
    /// accumulator; a forward-echelon one recovers nothing until complete.
    pub fn recovered(&self) -> impl Iterator<Item = (usize, &[u8])> + '_ {
        (0..self.cols).filter_map(move |column| {
            let physical = self.logical_to_physical(column);
            if !self.recovered[physical] {
                return None;
            }
            self.pivot_of_col[physical].map(|slot| (column, self.rhs.row(slot)))
        })
    }

    /// The recovered payload at `column`, if its retained row is a lone unit.
    #[must_use]
    pub fn recovered_value(&self, column: usize) -> Option<&[u8]> {
        let physical = self.logical_to_physical(column);
        if !self.recovered[physical] {
            return None;
        }
        self.pivot_of_col[physical].map(|slot| self.rhs.row(slot))
    }

    /// Logical columns that became recovered during the most recent
    /// absorption.
    ///
    /// The slice is cleared by the next [`Self::absorb`] or
    /// [`Self::advance_prefix`] call.
    #[must_use]
    pub fn newly_recovered_columns(&self) -> &[usize] {
        &self.newly_recovered
    }

    /// Variables that became recovered during the most recent absorption.
    ///
    /// The list is cleared by the next [`Self::absorb`] or
    /// [`Self::advance_prefix`] call.
    pub fn newly_recovered(&self) -> impl Iterator<Item = (usize, &[u8])> + '_ {
        self.newly_recovered.iter().filter_map(|&column| {
            let physical = self.logical_to_physical(column);
            self.pivot_of_col[physical].map(|slot| (column, self.rhs.row(slot)))
        })
    }
}

impl<F: FieldKernels> core::fmt::Debug for Echelon<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Echelon")
            .field("cols", &self.cols)
            .field("rhs_cols", &self.rhs_cols)
            .field("rank", &self.rank)
            .field("reduced", &self.reduced)
            .finish_non_exhaustive()
    }
}
