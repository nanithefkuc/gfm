//! Error types, one small enum per failure domain.
//!
//! Hand-rolled, matching the rest of the stack. Struct-variant fields carry
//! both the offending value and the limit it violated, so a caller can
//! report or recover without re-running the check.
//!
//! Note what is *not* an error: rank deficiency, a zero determinant, an
//! empty kernel, and division by zero (`inv(0) == 0` is inherited from
//! `fgf` and is total). Only invalid geometry and an inconsistent system
//! are errors.

use core::fmt;

/// Invalid matrix geometry at a public boundary.
///
/// Geometry is validated with `checked_mul` before any mutation begins, so
/// receiving this error guarantees the inputs were left untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometryError {
    /// A dimension product overflowed `usize` (`rows * pitch`, or `cols`
    /// times the element width).
    Overflow {
        /// The first operand of the product that overflowed.
        rows: usize,
        /// The second operand of the product that overflowed.
        pitch: usize,
    },
    /// Buffer length is not a whole number of elements.
    Ragged {
        /// The buffer length in bytes that was rejected.
        len: usize,
        /// The element size in bytes it must be a multiple of.
        element_bytes: usize,
    },
    /// Operand shapes do not compose.
    Shape {
        /// `(rows, cols)` of the left operand.
        lhs: (usize, usize),
        /// `(rows, cols)` of the right operand.
        rhs: (usize, usize),
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Overflow { rows, pitch } => {
                write!(f, "matrix geometry overflows usize: {rows} * {pitch}")
            }
            Self::Ragged { len, element_bytes } => write!(
                f,
                "buffer of {len} bytes is not a whole number of {element_bytes}-byte elements"
            ),
            Self::Shape { lhs, rhs } => {
                write!(f, "operand shapes do not compose: {lhs:?} vs {rhs:?}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GeometryError {}

/// A system could not be solved as posed.
///
/// Rank deficiency alone is *not* an error: a rank-deficient matrix produces
/// a rank-deficient answer. These are the two failures that remain.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveError {
    /// A zero row with a nonzero right-hand side: the system has no solution.
    Inconsistent {
        /// Index of a genuinely inconsistent row.
        row: usize,
    },
    /// An inverse was requested of a square matrix that is not full rank.
    Singular {
        /// The rank the factorization actually found.
        rank: usize,
        /// The order (side length) a full-rank matrix would need.
        order: usize,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Inconsistent { row } => {
                write!(f, "system is inconsistent at row {row}")
            }
            Self::Singular { rank, order } => {
                write!(f, "matrix is singular: rank {rank} of order {order}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SolveError {}
