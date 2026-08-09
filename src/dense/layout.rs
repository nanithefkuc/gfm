//! Pitch, alignment, and zero-padding policy for the dense domain.
//!
//! The three layout properties are type invariants of [`crate::Matrix`], not
//! conventions:
//!
//! 1. the base of the row region is [`ALIGN`]-byte aligned;
//! 2. the pitch (bytes per row) is a multiple of [`ALIGN`];
//! 3. every padding byte — the region between the live row bytes and the
//!    pitch — is zero and stays zero.
//!
//! Together they let `fgf`'s kernels process whole lanes with no tail mask
//! and no branch, and let one aligned peel align every row of a group. The
//! measured value of these invariants is recorded in `BENCHMARKS.md`.

use crate::GeometryError;
use alloc::boxed::Box;
use alloc::vec;

/// Byte alignment of a matrix base and row pitch: one AVX2/GFNI lane.
pub const ALIGN: usize = 32;

/// The smallest multiple of [`ALIGN`] that holds `row_bytes` bytes.
///
/// # Errors
///
/// [`GeometryError::Overflow`] if rounding up would overflow `usize`.
pub fn pitch_for(row_bytes: usize) -> Result<usize, GeometryError> {
    let rounded = row_bytes
        .checked_add(ALIGN - 1)
        .ok_or(GeometryError::Overflow {
            rows: row_bytes,
            pitch: ALIGN,
        })?;
    Ok(rounded / ALIGN * ALIGN)
}

/// A zeroed heap buffer of at least `len` bytes plus the offset of its first
/// [`ALIGN`]-aligned byte.
///
/// The allocation is over-long by `ALIGN - 1` bytes so an aligned offset
/// always exists; the offset is computed from the address, never assumed.
pub(crate) fn aligned_zeroed(len: usize) -> (Box<[u8]>, usize) {
    let buf = vec![0u8; len + (ALIGN - 1)].into_boxed_slice();
    let addr = buf.as_ptr() as usize;
    let offset = (ALIGN - addr % ALIGN) % ALIGN;
    (buf, offset)
}
