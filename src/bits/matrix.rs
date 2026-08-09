//! The owning GF(2) container [`BitMatrix`].
//!
//! Rows are `u64`-packed bit strings, stored contiguously `pitch` bytes
//! apart with the base 64-byte aligned and every padding bit zero. Bit `c`
//! of logical row `r` is bit `c % 64` of word `map[r] * pitch_words + c /
//! 64`. As in the dense domain, row exchange is an index operation on the
//! row map; data moves only through [`BitMatrix::compact_rows`].
//!
//! Bits between `cols` and the next word boundary are padding too and are
//! always zero. That is why there is no `row_mut`: mutable word access could
//! set them, and the elimination's `trailing_zeros` pivot search reads whole
//! words. Mutation goes through [`BitMatrix::set`], which cannot address
//! them.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::GeometryError;
use crate::dense::{Perm, invert_in_place};

/// Byte alignment of a bit-matrix base and row pitch: one 64-byte line.
pub const ALIGN: usize = 64;

/// Words per aligned pitch unit: [`ALIGN`] bytes is eight `u64` words.
const WORDS_PER_ALIGN: usize = ALIGN / 8;

/// A dense, row-major bit matrix, owning its storage.
pub struct BitMatrix {
    buf: Box<[u64]>,
    /// Offset of the first 64-byte-aligned word within `buf`.
    off: usize,
    rows: usize,
    /// Columns, in bits.
    cols: usize,
    /// Words per row: a multiple of [`WORDS_PER_ALIGN`].
    pitch_words: usize,
    /// Logical-to-physical row indirection.
    map: Vec<usize>,
}

impl BitMatrix {
    /// Validates the geometry and returns `(row_words, pitch_words)`.
    fn geometry(rows: usize, cols: usize) -> Result<(usize, usize), GeometryError> {
        let row_words = cols.checked_add(63).ok_or(GeometryError::Overflow {
            rows: cols,
            pitch: 64,
        })? / 64;
        let pitch_words =
            row_words
                .checked_add(WORDS_PER_ALIGN - 1)
                .ok_or(GeometryError::Overflow {
                    rows: row_words,
                    pitch: WORDS_PER_ALIGN,
                })?
                / WORDS_PER_ALIGN
                * WORDS_PER_ALIGN;
        rows.checked_mul(pitch_words)
            .ok_or(GeometryError::Overflow {
                rows,
                pitch: pitch_words,
            })?;
        Ok((row_words, pitch_words))
    }

    /// Creates a `rows` by `cols` matrix of zero bits.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Overflow`] if the geometry overflows `usize`.
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, GeometryError> {
        let (_, pitch_words) = Self::geometry(rows, cols)?;
        let (buf, off) = aligned_zeroed_words(rows * pitch_words);
        Ok(Self {
            buf,
            off,
            rows,
            cols,
            pitch_words,
            map: (0..rows).collect(),
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
            m.set(i, i, true);
        }
        Ok(m)
    }

    /// Creates a matrix from packed words: `rows * ceil(cols / 64)` words,
    /// row-major. Bits beyond `cols` in the last word of each row are
    /// cleared on entry, keeping the padding zero.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Shape`] if the word count does not match the declared
    /// shape, [`GeometryError::Overflow`] if the geometry overflows `usize`.
    pub fn from_rows(rows: usize, cols: usize, data: &[u64]) -> Result<Self, GeometryError> {
        let (row_words, _) = Self::geometry(rows, cols)?;
        let expected = rows * row_words;
        if data.len() != expected {
            return Err(GeometryError::Shape {
                lhs: (rows, cols),
                rhs: (data.len(), 1),
            });
        }
        let mut m = Self::zeros(rows, cols)?;
        let mask = live_mask(cols);
        for r in 0..rows {
            let start = m.map[r] * m.pitch_words;
            m.region_mut()[start..start + row_words]
                .copy_from_slice(&data[r * row_words..(r + 1) * row_words]);
            if row_words > 0 {
                m.region_mut()[start + row_words - 1] &= mask;
            }
        }
        Ok(m)
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns, in bits.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes per row: a multiple of [`ALIGN`] and at least `ceil(cols / 64) * 8`.
    #[must_use]
    pub const fn pitch(&self) -> usize {
        self.pitch_words * 8
    }

    /// Live words per row: `ceil(cols / 64)`.
    #[must_use]
    pub const fn row_words(&self) -> usize {
        self.cols.div_ceil(64)
    }

    /// Returns `true` if the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The aligned, pitched physical backing region, `rows * pitch_words` words.
    pub(crate) fn region(&self) -> &[u64] {
        &self.buf[self.off..self.off + self.rows * self.pitch_words]
    }

    /// The mutable aligned, pitched physical backing region.
    pub(crate) fn region_mut(&mut self) -> &mut [u64] {
        &mut self.buf[self.off..self.off + self.rows * self.pitch_words]
    }

    /// The bit at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> bool {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let word = self.region()[self.map[row] * self.pitch_words + col / 64];
        word & (1 << (col % 64)) != 0
    }

    /// Sets the bit at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: bool) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let start = self.map[row] * self.pitch_words + col / 64;
        let word = &mut self.region_mut()[start];
        if value {
            *word |= 1 << (col % 64);
        } else {
            *word &= !(1 << (col % 64));
        }
    }

    /// Borrows logical row `r`'s live words: `ceil(cols / 64)` of them, with
    /// the bits beyond `cols` in the last word always zero.
    ///
    /// # Panics
    ///
    /// Panics if `r` is out of bounds.
    #[must_use]
    pub fn row(&self, r: usize) -> &[u64] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch_words;
        &self.region()[start..start + self.row_words()]
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

    /// Applies `p` to the row order, as an index operation on the row map.
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
    /// Applies the inverse of `p` to the row order, as an index operation.
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
    /// the row map to the identity. This is the one place row data moves.
    ///
    /// Allocation-free, like the dense domain's: the map is inverted in
    /// place, then the inverse drives the standard in-place scatter.
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

    /// Exchanges the full pitched storage of physical rows `a` and `b`.
    fn swap_pitched_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let pitch_words = self.pitch_words;
        let region = self.region_mut();
        let (head, tail) = region.split_at_mut(hi * pitch_words);
        head[lo * pitch_words..(lo + 1) * pitch_words].swap_with_slice(&mut tail[..pitch_words]);
    }

    /// Disjoint mutable borrows of the live words of logical rows `a` and
    /// `b`, returned in argument order. The bit analogue of the dense
    /// domain's `two_live_rows`.
    ///
    /// # Panics
    ///
    /// Panics if `a == b` or either index is out of bounds.
    pub(crate) fn two_live_rows(&mut self, a: usize, b: usize) -> (&mut [u64], &mut [u64]) {
        assert!(a != b, "rows must differ");
        assert!(a < self.rows && b < self.rows, "row index out of bounds");
        let (pa, pb) = (self.map[a], self.map[b]);
        let (lo, hi) = if pa < pb { (pa, pb) } else { (pb, pa) };
        let pitch = self.pitch_words;
        let live = self.row_words();
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

    /// Mutably borrows logical row `r`'s live words. In-crate callers must
    /// leave the padding bits of the last word zero; the elimination's range
    /// helpers do, and never write beyond `cols`.
    pub(crate) fn live_row_mut(&mut self, r: usize) -> &mut [u64] {
        assert!(r < self.rows, "row index out of bounds");
        let start = self.map[r] * self.pitch_words;
        let live = self.row_words();
        &mut self.region_mut()[start..start + live]
    }

    /// Zeros the live words of logical row `r`.
    pub(crate) fn zero_row(&mut self, r: usize) {
        self.live_row_mut(r).fill(0);
    }

    /// Whether logical row `r` is entirely zero.
    pub(crate) fn row_is_zero(&self, r: usize) -> bool {
        self.row(r).iter().all(|&w| w == 0)
    }

    /// Copies the live words of `src`'s logical row `src_row` into `self`'s
    /// logical row `dst_row`. The two matrices must have equal row width.
    pub(crate) fn copy_row_from(&mut self, dst_row: usize, src: &BitMatrix, src_row: usize) {
        let words = src.row(src_row);
        self.live_row_mut(dst_row).copy_from_slice(words);
    }

    /// Exchanges bit columns `c0` and `c1` across every physical row. Unlike
    /// rows, columns have no indirection: the data moves, one bit swap per
    /// row. Both columns are live, so no padding bit is ever touched.
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
        let (w0, b0) = (c0 / 64, c0 % 64);
        let (w1, b1) = (c1 / 64, c1 % 64);
        let pitch = self.pitch_words;
        let rows = self.rows;
        let region = self.region_mut();
        for r in 0..rows {
            let base = r * pitch;
            let bit0 = (region[base + w0] >> b0) & 1;
            let bit1 = (region[base + w1] >> b1) & 1;
            if bit0 != bit1 {
                region[base + w0] ^= 1 << b0;
                region[base + w1] ^= 1 << b1;
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
}

/// The mask of live bits in the last word of a row: `cols % 64` low bits,
/// or all bits when `cols` is a multiple of 64.
fn live_mask(cols: usize) -> u64 {
    if cols.is_multiple_of(64) {
        u64::MAX
    } else {
        (1 << (cols % 64)) - 1
    }
}

/// A zeroed heap buffer of at least `words` `u64` words plus the offset, in
/// words, of its first 64-byte-aligned word.
fn aligned_zeroed_words(words: usize) -> (Box<[u64]>, usize) {
    let buf = vec![0u64; words + (WORDS_PER_ALIGN - 1)].into_boxed_slice();
    let addr = buf.as_ptr() as usize;
    let offset = (ALIGN - addr % ALIGN) % ALIGN / 8;
    (buf, offset)
}

impl Clone for BitMatrix {
    fn clone(&self) -> Self {
        // Re-align the fresh allocation rather than copy the offset.
        let total = self.rows * self.pitch_words;
        let (mut buf, off) = aligned_zeroed_words(total);
        buf[off..off + total].copy_from_slice(self.region());
        Self {
            buf,
            off,
            rows: self.rows,
            cols: self.cols,
            pitch_words: self.pitch_words,
            map: self.map.clone(),
        }
    }
}

impl PartialEq for BitMatrix {
    /// Logical content equality: shape and live words in logical row order.
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && (0..self.rows).all(|r| self.row(r) == other.row(r))
    }
}

impl Eq for BitMatrix {}

impl fmt::Debug for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("pitch", &self.pitch())
            .finish_non_exhaustive()
    }
}

/// Unstable inspection API, available only with feature `internals`.
#[cfg(feature = "internals")]
impl BitMatrix {
    /// Address of the first word of the physical backing region. A multiple
    /// of [`ALIGN`] by construction.
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

    /// The whole physical backing region, padding included: `rows *
    /// pitch_words` words, laid out as physical rows.
    #[must_use]
    pub fn pitched_buffer(&self) -> &[u64] {
        self.region()
    }
}
