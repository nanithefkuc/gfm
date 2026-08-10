//! Error types, one small enum per failure domain.
//!
//! Hand-rolled, matching the rest of the stack. Struct-variant fields carry
//! both the offending value and the limit it violated, so a caller can
//! report or recover without re-running the check.
//!
//! Note what is *not* an error: rank deficiency, a zero determinant, an
//! empty kernel, and division by zero (`inv(0) == 0` is inherited from
//! `fgf` and is total). Only invalid geometry, an inconsistent system, and
//! a reduction that fails to terminate are errors.

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
    /// A structured matrix asked for more distinct points than the field
    /// holds: `requested` points over a field of `order` elements.
    Capacity {
        /// Number of distinct field points the construction needs.
        requested: usize,
        /// The field's element count.
        order: u128,
    },
    /// Two structured-matrix points coincide — a repeated evaluation point,
    /// or overlapping Cauchy index sets `X ∩ Y ≠ ∅` — which makes the
    /// construction singular or undefined.
    Collision {
        /// The raw byte value of the point that recurred.
        value: u64,
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
            Self::Capacity { requested, order } => write!(
                f,
                "structured matrix needs {requested} distinct points over a field of {order}"
            ),
            Self::Collision { value } => {
                write!(
                    f,
                    "structured-matrix point {value:#x} is repeated or not disjoint"
                )
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
    /// An inverse was requested of a matrix that is not square or not full
    /// rank.
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

/// A polynomial-matrix reduction failed to terminate within its ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceError {
    /// The reduction exceeded its iteration ceiling.
    Diverged {
        /// Iterations actually performed.
        iterations: usize,
        /// The ceiling that was exceeded.
        ceiling: usize,
    },
    /// A row exposes a different number of polynomial columns than the shift.
    ShiftCount {
        /// Polynomial columns in the row.
        columns: usize,
        /// Entries in the shift vector.
        shifts: usize,
    },
    /// Adding a polynomial degree and its shift overflowed `usize`.
    DegreeOverflow {
        /// Unshifted polynomial degree.
        degree: usize,
        /// Shift assigned to that polynomial column.
        shift: usize,
    },
    /// Storage for the leading-row schedule could not be reserved.
    AllocationFailed {
        /// Number of schedule entries requested.
        entries: usize,
    },
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Diverged {
                iterations,
                ceiling,
            } => write!(
                f,
                "reduction diverged: {iterations} iterations exceeded the ceiling of {ceiling}"
            ),
            Self::ShiftCount { columns, shifts } => write!(
                f,
                "polynomial row has {columns} columns but the shift has {shifts} entries"
            ),
            Self::DegreeOverflow { degree, shift } => write!(
                f,
                "shifted polynomial degree overflows usize: {degree} + {shift}"
            ),
            Self::AllocationFailed { entries } => {
                write!(
                    f,
                    "failed to reserve {entries} polynomial reduction entries"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReduceError {}
