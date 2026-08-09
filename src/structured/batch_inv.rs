//! Montgomery batch inversion, generic over the field.
//!
//! Inverting `n` elements one at a time is `n` field inversions. Montgomery's
//! trick trades all but one of them for multiplications: form the running
//! product, invert it once, then walk backwards peeling one factor at a time —
//! one inversion and `3(n − 1)` multiplications total.
//!
//! Zero is handled by the field's own total convention (`inv(0) == 0`, I3): a
//! zero input is skipped in the product and left as zero on the way out, so a
//! buffer with zeros still inverts correctly rather than collapsing the whole
//! product to zero.

use alloc::vec;

use fgf::FieldKernels;
use fgf::field::Elem;

/// Inverts every element of `values` in place, mapping zero to zero.
///
/// Equivalent to calling [`Elem::inv`] on each element, but with a single
/// field inversion regardless of `n`.
pub fn batch_invert<F: FieldKernels>(values: &mut [F::Elem]) {
    let n = values.len();
    if n == 0 {
        return;
    }
    // `prefix[i]` is the product of the nonzero elements strictly before `i`.
    let mut prefix = vec![F::Elem::ONE; n];
    let mut acc = F::Elem::ONE;
    for i in 0..n {
        prefix[i] = acc;
        if !values[i].is_zero() {
            acc = acc.mul(values[i]);
        }
    }
    // `acc` is now the product of every nonzero element; invert it once and
    // peel factors off from the back.
    let mut inv_acc = acc.inv();
    for i in (0..n).rev() {
        if values[i].is_zero() {
            continue; // inv(0) == 0, already stored.
        }
        let original = values[i];
        values[i] = inv_acc.mul(prefix[i]);
        inv_acc = inv_acc.mul(original);
    }
}
