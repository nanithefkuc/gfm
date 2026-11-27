//! Consumer-owned equation sources for [`Hybrid`](super::Hybrid).

use fgf::{Elem, FieldBuffer};

/// One borrowed equation supplied through [`SparseEquationSource`].
///
/// Supports are sorted, strictly increasing column indices. Binary equations
/// imply a unit coefficient at every support position; field equations carry
/// parallel coefficients. The right-hand side uses the same packed
/// field-element representation as [`crate::Matrix`].
#[derive(Clone, Copy, Debug)]
pub enum SparseEquation<'a, F: FieldBuffer> {
    /// A binary equation whose support coefficients are all one.
    Binary {
        /// Sorted, distinct column indices.
        support: &'a [u32],
        /// Packed right-hand side.
        rhs: &'a [u8],
    },
    /// A field-valued equation.
    Field {
        /// Sorted, distinct column indices.
        support: &'a [u32],
        /// Coefficients parallel to `support`.
        coeffs: &'a [Elem<F>],
        /// Packed right-hand side.
        rhs: &'a [u8],
    },
}

/// Source of borrowed equations for a [`Hybrid`](super::Hybrid) system.
///
/// `gfm` owns this seam because it owns the solver accepting the equations. A
/// graph crate or codec implements the trait for its local
/// residual-equation container; neither `gfm` nor the trait implementation
/// needs an inverse dependency. Implementations must preserve equation order
/// when calling `visit`.
pub trait SparseEquationSource<F: FieldBuffer> {
    /// Visits every admitted equation exactly once, in source order.
    fn for_each_equation(&self, visit: &mut impl FnMut(SparseEquation<'_, F>));
}
