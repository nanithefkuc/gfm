//! Triangular solves, built on `fgf::ops` row kernels.
//!
//! [`solve_lower_unit_into`] is the forward substitution for a unit lower
//! triangle; [`solve_upper_into`] is the back substitution for an upper
//! triangle with a nonzero diagonal. Both run in place on the right-hand
//! side: the update `row_i ^= f · row_t` is one [`ops::mul_add`], so the
//! arithmetic is exactly the unblocked elimination's, term for term.

use fgf::{FieldKernels, field::Elem, ops};

use crate::dense::Matrix;

/// Solves `L·X = B` in place on `b`, where `l` is unit lower triangular:
/// its diagonal is implicitly one, entries above the diagonal are ignored,
/// and `l[(i, t)]` for `t < i` is the factor.
///
/// # Panics
///
/// Panics unless `l` is `k × k` and `b` is `k × s`.
pub fn solve_lower_unit_into<F: FieldKernels>(l: &Matrix<F>, b: &mut Matrix<F>) {
    assert_eq!(l.rows(), l.cols(), "lower triangle must be square");
    assert_eq!(b.rows(), l.rows(), "right-hand side row mismatch");
    for i in 0..l.rows() {
        for t in 0..i {
            let f = l.get(i, t);
            if f.is_zero() {
                continue;
            }
            let (row_i, row_t) = b.two_live_rows(i, t);
            crate::row_ops::mul_add::<F>(row_i, f, row_t);
        }
    }
}

/// Solves `U·X = B` in place on `b`, where `u` is upper triangular with a
/// nonzero diagonal: `u[(i, t)]` for `t >= i` is the entry, entries below
/// the diagonal are ignored.
///
/// # Panics
///
/// Panics unless `u` is `k × k` and `b` is `k × s`, or if a diagonal entry
/// of `u` is zero (a singular triangle is a contract violation).
pub fn solve_upper_into<F: FieldKernels>(u: &Matrix<F>, b: &mut Matrix<F>) {
    assert_eq!(u.rows(), u.cols(), "upper triangle must be square");
    assert_eq!(b.rows(), u.rows(), "right-hand side row mismatch");
    for i in (0..u.rows()).rev() {
        for t in (i + 1)..u.rows() {
            let f = u.get(i, t);
            if f.is_zero() {
                continue;
            }
            let (row_i, row_t) = b.two_live_rows(i, t);
            crate::row_ops::mul_add::<F>(row_i, f, row_t);
        }
        let d = u.get(i, i);
        assert!(!d.is_zero(), "upper triangle has a zero diagonal entry");
        if !d.is_one() {
            ops::mul_assign::<F>(b.row_mut(i), d.inv());
        }
    }
}
