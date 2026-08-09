//! The owning dense container [`Matrix<F>`] and its borrow views.
//!
//! Rows are stored contiguously in one buffer, `pitch` bytes apart, with the
//! base [`crate::dense::layout::ALIGN`]-aligned and every padding byte zero (the layout invariants
//! in [`crate::dense::layout`]). A logical-to-physical row map rides beside
//! the buffer: exchanging rows is an index operation on the map, and the data
//! does not move unless contiguous physical rows are required — then it moves
//! once, through [`Matrix::compact_rows`].
//!
//! Views ([`View`], [`ViewMut`]) are cheap re-borrows of the same state, the
//! borrow-only counterpart of the owning container.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use fgf::{FieldKernels, field::Elem};

use crate::GeometryError;
use crate::dense::layout::{aligned_zeroed, pitch_for};
use crate::dense::{Perm, invert_in_place};

/// A dense, row-major matrix over GF(2^m), owning its storage.
///
/// Element `(r, c)` lives at `map[r] * pitch + c * F::BYTES` of the buffer,
/// little-endian. The invariants of [`crate::dense::layout`] hold from
/// construction through every mutating operation.
pub struct Matrix<F: FieldKernels> {
    buf: Box<[u8]>,
    /// Offset of the first aligned byte within `buf`.
    off: usize,
    rows: usize,
    cols: usize,
    /// Bytes per row: a multiple of [`crate::dense::layout::ALIGN`] and at least `cols * F::BYTES`.
    pitch: usize,
    /// Logical-to-physical row indirection; the data never moves for a row
    /// exchange, this map does.
    map: Vec<usize>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Matrix<F> {
    /// Validates the geometry and returns `(cols_bytes, pitch)`.
    fn geometry(rows: usize, cols: usize) -> Result<(usize, usize), GeometryError> {
        let cols_bytes = cols.checked_mul(F::BYTES).ok_or(GeometryError::Overflow {
            rows: cols,
            pitch: F::BYTES,
        })?;
        let pitch = pitch_for(cols_bytes)?;
        rows.checked_mul(pitch)
            .ok_or(GeometryError::Overflow { rows, pitch })?;
        Ok((cols_bytes, pitch))
    }

    /// Creates a `rows` by `cols` matrix of zeros.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Overflow`] if the geometry overflows `usize`.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, GeometryError> {
        let (_, pitch) = Self::geometry(rows, cols)?;
        let (buf, off) = aligned_zeroed(rows * pitch);
        Ok(Self {
            buf,
            off,
            rows,
            cols,
            pitch,
            map: (0..rows).collect(),
            field: PhantomData,
        })
    }

    /// Creates the `n` by `n` identity matrix.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Overflow`] if the geometry overflows `usize`.
    pub fn identity(n: usize) -> Result<Self, GeometryError> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.set(i, i, F::Elem::ONE);
        }
        Ok(m)
    }

    /// Creates a matrix from packed row-major data: `rows * cols` elements,
    /// little-endian, `rows * cols * F::BYTES` bytes in total.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Ragged`] if `data` is not a whole number of elements,
    /// [`GeometryError::Shape`] if its length does not match the declared
    /// shape, [`GeometryError::Overflow`] if the geometry overflows `usize`.
    pub fn from_rows(rows: usize, cols: usize, data: &[u8]) -> Result<Self, GeometryError> {
        if !data.len().is_multiple_of(F::BYTES) {
            return Err(GeometryError::Ragged {
                len: data.len(),
                element_bytes: F::BYTES,
            });
        }
        let (cols_bytes, _) = Self::geometry(rows, cols)?;
        // `rows * cols_bytes <= rows * pitch`, already checked above.
        let expected = rows * cols_bytes;
        if data.len() != expected {
            return Err(GeometryError::Shape {
                lhs: (rows, cols),
                rhs: (data.len() / F::BYTES, 1),
            });
        }
        let mut m = Self::zeros(rows, cols)?;
        for r in 0..rows {
            m.row_mut(r)
                .copy_from_slice(&data[r * cols_bytes..(r + 1) * cols_bytes]);
        }
        Ok(m)
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes per row: a multiple of [`crate::dense::layout::ALIGN`] and at least `cols * F::BYTES`.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// Returns `true` if the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The aligned, pitched physical backing region, `rows * pitch` bytes.
    pub(crate) fn region(&self) -> &[u8] {
        &self.buf[self.off..self.off + self.rows * self.pitch]
    }

    /// The mutable aligned, pitched physical backing region.
    pub(crate) fn region_mut(&mut self) -> &mut [u8] {
        &mut self.buf[self.off..self.off + self.rows * self.pitch]
    }

    /// Live bytes of a physical row: padding excluded.
    pub(crate) fn live_bytes(&self) -> usize {
        self.cols * F::BYTES
    }

    /// The element at `(row, col)`, decoded from its little-endian bytes.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> F::Elem {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch + col * F::BYTES;
        F::read(&self.region()[start..start + F::BYTES])
    }

    /// Sets the element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: F::Elem) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch + col * F::BYTES;
        F::write(&mut self.region_mut()[start..start + F::BYTES], value);
    }

    /// Borrows logical row `r`: the `cols * F::BYTES` live bytes. Padding is
    /// unreachable through this slice, which is how it stays zero.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u8] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch;
        &self.region()[start..start + self.live_bytes()]
    }

    /// Mutably borrows logical row `r`'s live bytes.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row_mut(&mut self, r: usize) -> &mut [u8] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch;
        let live = self.live_bytes();
        &mut self.region_mut()[start..start + live]
    }

    /// Exchanges logical rows `a` and `b`. A no-op when they are the same
    /// row. This is an index operation on the row map: the data does not
    /// move.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(a < self.rows && b < self.rows, "row index out of bounds");
        self.map.swap(a, b);
    }

    /// Applies `p` to the row order: afterwards, logical row `r` is what was
    /// logical row `p`'s image of `r` before. An index operation on the row
    /// map; the data does not move.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Shape`] if `p` does not act on `rows` positions.
    /// The matrix is left unchanged in that case.
    pub fn apply_row_perm(&mut self, p: &Perm) -> Result<(), GeometryError> {
        if p.len() != self.rows {
            return Err(GeometryError::Shape {
                lhs: (self.rows, self.cols),
                rhs: (p.len(), p.len()),
            });
        }
        p.apply(&mut self.map);
        Ok(())
    }
    /// Applies the inverse of `p` to the row order, without moving row data.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Shape`] if `p` does not act on `rows` positions.
    /// The matrix is left unchanged in that case.
    pub fn apply_row_perm_inv(&mut self, p: &Perm) -> Result<(), GeometryError> {
        if p.len() != self.rows {
            return Err(GeometryError::Shape {
                lhs: (self.rows, self.cols),
                rhs: (p.len(), p.len()),
            });
        }
        p.apply_inv(&mut self.map);
        Ok(())
    }

    /// Physically reorders the rows to match the logical order and resets
    /// the row map to the identity. This is the one place row data moves;
    /// every permuting operation before it was an index edit.
    ///
    /// Allocation-free: the map is inverted in place (high-bit tagging —
    /// row counts are far below `usize::MAX / 2` by geometry), then the
    /// inverse drives the standard in-place scatter, whose side effect is to
    /// leave the map as the identity.
    pub fn compact_rows(&mut self) {
        let n = self.rows;
        invert_in_place(&mut self.map);
        for i in 0..n {
            while self.map[i] != i {
                let j = self.map[i];
                self.swap_pitched_rows(i, j);
                self.map.swap(i, j);
            }
        }
    }

    /// Disjoint mutable borrows of the live regions of logical rows `a` and
    /// `b`, returned in argument order.
    ///
    /// # Panics
    ///
    /// Panics if `a == b` or either is out of bounds.
    pub(crate) fn two_live_rows(&mut self, a: usize, b: usize) -> (&mut [u8], &mut [u8]) {
        assert!(a != b, "rows must differ");
        assert!(a < self.rows && b < self.rows, "row index out of bounds");
        let (pa, pb) = (self.map[a], self.map[b]);
        let (lo, hi) = if pa < pb { (pa, pb) } else { (pb, pa) };
        let pitch = self.pitch;
        let live = self.live_bytes();
        let region = self.region_mut();
        let (head, tail) = region.split_at_mut(hi * pitch);
        let row_lo = &mut head[lo * pitch..lo * pitch + live];
        let row_hi = &mut tail[..live];
        if pa < pb {
            (row_lo, row_hi)
        } else {
            (row_hi, row_lo)
        }
    }

    /// Exchanges columns `c0` and `c1` in place. Unlike rows, columns have no
    /// indirection: the data moves, one element swap per row.
    ///
    /// # Panics
    ///
    /// Panics if either column is out of bounds.
    pub(crate) fn swap_cols(&mut self, c0: usize, c1: usize) {
        assert!(
            c0 < self.cols && c1 < self.cols,
            "column index out of bounds"
        );
        if c0 == c1 {
            return;
        }
        let (b0, b1) = (c0 * F::BYTES, c1 * F::BYTES);
        let pitch = self.pitch;
        let region = self.region_mut();
        for row in region.chunks_exact_mut(pitch) {
            for w in 0..F::BYTES {
                row.swap(b0 + w, b1 + w);
            }
        }
    }

    /// Undoes a column permutation recorded as a swap list: applies the
    /// swaps in reverse order. The inverse of applying `q` to the columns.
    pub(crate) fn apply_col_perm_inv(&mut self, q: &Perm) {
        for (i, &j) in q.swaps().iter().enumerate().rev() {
            if i != j {
                self.swap_cols(i, j);
            }
        }
    }

    /// Exchanges the full pitched storage of physical rows `a` and `b`,
    /// padding included (both are zero there, and stay zero).
    fn swap_pitched_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let pitch = self.pitch;
        let region = self.region_mut();
        let (head, tail) = region.split_at_mut(hi * pitch);
        head[lo * pitch..(lo + 1) * pitch].swap_with_slice(&mut tail[..pitch]);
    }

    /// Borrows the whole matrix as a read-only view.
    #[must_use]
    pub fn as_view(&self) -> View<'_, F> {
        View {
            data: self.region(),
            map: &self.map,
            rows: self.rows,
            cols: self.cols,
            pitch: self.pitch,
            field: PhantomData,
        }
    }

    /// Borrows the whole matrix as a mutable view.
    #[must_use]
    pub fn as_view_mut(&mut self) -> ViewMut<'_, F> {
        // Destructure for disjoint field borrows: the data slice and the row
        // map come from different fields of the same `&mut self`.
        let Self {
            buf,
            off,
            rows,
            cols,
            pitch,
            map,
            ..
        } = self;
        ViewMut {
            data: &mut buf[*off..*off + *rows * *pitch],
            map,
            rows: *rows,
            cols: *cols,
            pitch: *pitch,
            field: PhantomData,
        }
    }
}

impl<F: FieldKernels> Clone for Matrix<F> {
    fn clone(&self) -> Self {
        // The clone gets its own allocation, whose base alignment differs in
        // general; re-align rather than copy the offset.
        let total = self.rows * self.pitch;
        let (mut buf, off) = aligned_zeroed(total);
        buf[off..off + total].copy_from_slice(self.region());
        Self {
            buf,
            off,
            rows: self.rows,
            cols: self.cols,
            pitch: self.pitch,
            map: self.map.clone(),
            field: PhantomData,
        }
    }
}

impl<F: FieldKernels> PartialEq for Matrix<F> {
    /// Logical content equality: shape and live bytes in logical row order.
    /// Physical arrangement (row map, padding, allocation offset) is
    /// incidental and not compared.
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && (0..self.rows).all(|r| self.row(r) == other.row(r))
    }
}

impl<F: FieldKernels> Eq for Matrix<F> {}

impl<F: FieldKernels> fmt::Debug for Matrix<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("field", &F::NAME)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("pitch", &self.pitch)
            .finish_non_exhaustive()
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl<F: FieldKernels> Matrix<F> {
    /// Address of the first byte of the physical backing region. A multiple
    /// of [`crate::dense::layout::ALIGN`] by construction.
    #[must_use]
    pub fn base_addr(&self) -> usize {
        self.region().as_ptr() as usize
    }

    /// The physical row logical row `r` currently maps to.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn physical_row_index(&self, r: usize) -> usize {
        assert!(r < self.rows, "row index out of bounds");
        self.map[r]
    }

    /// The whole physical backing region, padding included: `rows * pitch`
    /// bytes, laid out as physical rows of `pitch` bytes.
    #[must_use]
    pub fn pitched_buffer(&self) -> &[u8] {
        self.region()
    }
}

/// A read-only borrow of a [`Matrix`]'s rows, with its geometry and row map.
#[derive(Clone, Copy)]
pub struct View<'a, F: FieldKernels> {
    data: &'a [u8],
    map: &'a [usize],
    rows: usize,
    cols: usize,
    pitch: usize,
    field: PhantomData<F>,
}

impl<'a, F: FieldKernels> View<'a, F> {
    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes per row: a multiple of [`crate::dense::layout::ALIGN`] and at least `cols * F::BYTES`.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// The element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> F::Elem {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch + col * F::BYTES;
        F::read(&self.data[start..start + F::BYTES])
    }

    /// Borrows logical row `r`'s live bytes.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row(&self, r: usize) -> &'a [u8] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch;
        &self.data[start..start + self.cols * F::BYTES]
    }

    /// Splits the view vertically into `[top | bottom]` at row `at`,
    /// preserving logical row order. Either half may be empty.
    ///
    /// Returns `None` if `at > rows`.
    #[must_use]
    pub fn split_rows(&self, at: usize) -> Option<(View<'a, F>, View<'a, F>)> {
        if at > self.rows {
            return None;
        }
        let (top_map, bot_map) = self.map.split_at(at);
        let half = |rows: usize, map: &'a [usize]| View {
            data: self.data,
            map,
            rows,
            cols: self.cols,
            pitch: self.pitch,
            field: PhantomData,
        };
        Some((half(at, top_map), half(self.rows - at, bot_map)))
    }
}

impl<F: FieldKernels> fmt::Debug for View<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("View")
            .field("field", &F::NAME)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("pitch", &self.pitch)
            .finish_non_exhaustive()
    }
}

/// A mutable borrow of a [`Matrix`]'s rows, the write-capable counterpart of
/// [`View`].
pub struct ViewMut<'a, F: FieldKernels> {
    data: &'a mut [u8],
    map: &'a mut [usize],
    rows: usize,
    cols: usize,
    pitch: usize,
    field: PhantomData<F>,
}

impl<F: FieldKernels> ViewMut<'_, F> {
    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes per row: a multiple of [`crate::dense::layout::ALIGN`] and at least `cols * F::BYTES`.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch
    }

    /// The element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> F::Elem {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch + col * F::BYTES;
        F::read(&self.data[start..start + F::BYTES])
    }

    /// Sets the element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: F::Elem) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch + col * F::BYTES;
        F::write(&mut self.data[start..start + F::BYTES], value);
    }

    /// Borrows logical row `r`'s live bytes.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u8] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch;
        &self.data[start..start + self.cols * F::BYTES]
    }

    /// Mutably borrows logical row `r`'s live bytes.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row_mut(&mut self, r: usize) -> &mut [u8] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch;
        let live = self.cols * F::BYTES;
        &mut self.data[start..start + live]
    }

    /// Reborrows as an immutable [`View`].
    #[must_use]
    pub fn as_view(&self) -> View<'_, F> {
        View {
            data: self.data,
            map: self.map,
            rows: self.rows,
            cols: self.cols,
            pitch: self.pitch,
            field: PhantomData,
        }
    }

    /// Exchanges logical rows `a` and `b`. An index operation on the row
    /// map: the data does not move.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(a < self.rows && b < self.rows, "row index out of bounds");
        self.map.swap(a, b);
    }
}

impl<F: FieldKernels> fmt::Debug for ViewMut<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ViewMut")
            .field("field", &F::NAME)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("pitch", &self.pitch)
            .finish_non_exhaustive()
    }
}
