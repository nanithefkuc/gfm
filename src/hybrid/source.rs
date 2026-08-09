//! Consumer-owned row sources for [`Hybrid`](super::Hybrid).

use fgf::FieldKernels;

/// One borrowed equation supplied through [`DenseRows`].
///
/// Supports are sorted, strictly increasing column indices. Binary rows imply
/// a unit coefficient at every support position; field rows carry parallel
/// coefficients. The right-hand side uses the same packed field-element
/// representation as [`crate::Matrix`].
#[derive(Clone, Copy, Debug)]
pub enum DenseRow<'a, F: FieldKernels> {
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
        coeffs: &'a [F::Elem],
        /// Packed right-hand side.
        rhs: &'a [u8],
    },
}

/// Source of borrowed equations for a [`Hybrid`](super::Hybrid) system.
///
/// `gfm` owns this seam because it owns the solver accepting the rows. A graph
/// crate or codec implements the trait for its local residual-row container;
/// neither `gfm` nor the trait implementation needs an inverse dependency.
/// Implementations must preserve row order when calling `visit`.
pub trait DenseRows<F: FieldKernels> {
    /// Visits every admitted equation exactly once, in source order.
    fn for_each_row(&self, visit: &mut impl FnMut(DenseRow<'_, F>));
}
