//! Matrix multiply, built on `fgf::ops` row kernels.
//!
//! `C ^= A·B` one destination row at a time: each term is one
//! [`fgf::ops::mul_add`] with the coefficient read from `A`. Zero coefficients are
//! skipped before the kernel call, so the per-row cost scales with the
//! nonzero weight of `A`'s row, not its width. Every multiplication goes
//! through `fgf`; nothing here re-hosts field arithmetic.

use fgf::{FieldKernels, field::Elem};

use crate::dense::Matrix;

/// `out = A · B`.
///
/// Writes every cell of `out`; its previous contents do not matter.
///
/// # Panics
///
/// Panics unless `out` is `a.rows × b.cols` and `a.cols == b.rows`.
pub fn mul_into<F: FieldKernels>(out: &mut Matrix<F>, a: &Matrix<F>, b: &Matrix<F>) {
    check_shapes(out, a, b);
    for r in 0..out.rows() {
        out.row_mut(r).fill(0);
    }
    mul_add_into(out, a, b);
}

/// `C ^= A · B`, accumulating into `C`.
///
/// # Panics
///
/// Panics unless `c` is `a.rows × b.cols` and `a.cols == b.rows`.
pub fn mul_add_into<F: FieldKernels>(c: &mut Matrix<F>, a: &Matrix<F>, b: &Matrix<F>) {
    check_shapes(c, a, b);
    for i in 0..c.rows() {
        for t in 0..a.cols() {
            let f = a.get(i, t);
            if f.is_zero() {
                continue;
            }
            crate::row_ops::mul_add::<F>(c.row_mut(i), f, b.row(t));
        }
    }
}

fn check_shapes<F: FieldKernels>(c: &Matrix<F>, a: &Matrix<F>, b: &Matrix<F>) {
    assert_eq!(
        (c.rows(), c.cols()),
        (a.rows(), b.cols()),
        "matrix multiply shape mismatch",
    );
    assert_eq!(a.cols(), b.rows(), "matrix multiply shape mismatch");
}
