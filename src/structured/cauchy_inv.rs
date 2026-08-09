//! Closed-form inverse of a square Cauchy matrix, `O(k²)` and pivot-free.
//!
//! For `C[i][j] = 1/(x_i + y_j)` with the row set `X` and column set `Y`
//! disjoint, the inverse has the rational-Lagrange form
//!
//! ```text
//! R_i = ∏_h (x_i + y_h) / ∏_{l≠i} (x_i + x_l)
//! C_j = ∏_l (y_j + x_l) / ∏_{h≠j} (y_j + y_h)
//! C⁻¹[j][i] = R_i · C_j / (y_j + x_i).
//! ```
//!
//! The four set products cost `O(k²)`, every denominator is inverted in one
//! Montgomery batch, and emitting the inverse is `O(k²)` — so the whole
//! inverse is `O(k²)` with no elimination and no pivoting. In characteristic
//! two addition is subtraction, so no sign factors appear.
//!
//! Generalized from `srs/src/decoder/cauchy_inverse.rs`, which fixes the field
//! to GF(2^8); the algorithm and the zero-allocation scratch core are the
//! same.

use fgf::FieldKernels;
use fgf::field::Elem;

/// Scratch element count [`cauchy_inverse_coefficients_into`] needs for a
/// `k × k` matrix and any number of additional columns.
#[must_use]
pub const fn cauchy_scratch_len(k: usize) -> usize {
    4 * k + 2
}

/// Writes a Cauchy inverse and its products with additional Cauchy columns.
///
/// For the square matrix `A[i,j] = 1 / (row[i] + col[j])`, `inverse` receives
/// `A⁻¹` in row-major order. For every point `z` in `extra`, the corresponding
/// row of `extra_coefficients` receives `A⁻¹ · c(z)`, where
/// `c(z)[i] = 1 / (row[i] + z)`. This fused form lets erasure decoders cancel
/// already-known columns without materializing or multiplying by `A⁻¹`.
///
/// `row` and `col` must have equal length `k`, be distinct within each set,
/// and be disjoint. Every `extra` point must be disjoint from `row`.
/// `inverse` must have length `k²`, `extra_coefficients` length
/// `extra.len() * k`, and `scratch` at least [`cauchy_scratch_len`] elements.
///
/// # Panics
///
/// Panics when any required slice length is wrong.
pub fn cauchy_inverse_coefficients_into<F: FieldKernels>(
    row: &[F::Elem],
    col: &[F::Elem],
    extra: &[F::Elem],
    inverse: &mut [F::Elem],
    extra_coefficients: &mut [F::Elem],
    scratch: &mut [F::Elem],
) {
    let k = row.len();
    assert_eq!(col.len(), k, "Cauchy inverse shape mismatch");
    assert_eq!(inverse.len(), k * k, "Cauchy inverse output length");
    assert_eq!(
        extra_coefficients.len(),
        extra.len() * k,
        "Cauchy extra-coefficient output length"
    );
    assert!(
        scratch.len() >= cauchy_scratch_len(k),
        "Cauchy scratch is too short"
    );
    if k == 0 {
        return;
    }

    let (row_factors, rest) = scratch.split_at_mut(k);
    let (col_factors, rest) = rest.split_at_mut(k);
    let (prefix, suffix) = rest.split_at_mut(k + 1);
    let suffix = &mut suffix[..=k];

    // R_i = ∏_h (x_i + y_h) / ∏_{l≠i} (x_i + x_l).
    for (i, &x) in row.iter().enumerate() {
        let cross = col.iter().fold(F::Elem::ONE, |acc, &y| acc.mul(x.add(y)));
        let within = row
            .iter()
            .enumerate()
            .filter(|&(l, _)| l != i)
            .fold(F::Elem::ONE, |acc, (_, &other)| acc.mul(x.add(other)));
        row_factors[i] = cross.mul(within.inv());
    }
    // C_j = ∏_l (y_j + x_l) / ∏_{h≠j} (y_j + y_h).
    for (j, &y) in col.iter().enumerate() {
        let cross = row.iter().fold(F::Elem::ONE, |acc, &x| acc.mul(y.add(x)));
        let within = col
            .iter()
            .enumerate()
            .filter(|&(h, _)| h != j)
            .fold(F::Elem::ONE, |acc, (_, &other)| acc.mul(y.add(other)));
        col_factors[j] = cross.mul(within.inv());
    }
    // A⁻¹[j][i] = R_i · C_j / (y_j + x_i).
    for (j, &y) in col.iter().enumerate() {
        for (i, &x) in row.iter().enumerate() {
            inverse[j * k + i] = row_factors[i].mul(col_factors[j]).mul(y.add(x).inv());
        }
    }

    // A⁻¹ · c(z): C_j · ∏_{h≠j}(z + y_h) / ∏_l(z + x_l).
    for (extra_pos, &z) in extra.iter().enumerate() {
        let row_product = row.iter().fold(F::Elem::ONE, |acc, &x| acc.mul(z.add(x)));
        let row_product_inv = row_product.inv();
        prefix[0] = F::Elem::ONE;
        suffix[k] = F::Elem::ONE;
        for j in 0..k {
            prefix[j + 1] = prefix[j].mul(z.add(col[j]));
        }
        for j in (0..k).rev() {
            suffix[j] = suffix[j + 1].mul(z.add(col[j]));
        }
        for j in 0..k {
            extra_coefficients[extra_pos * k + j] = col_factors[j]
                .mul(prefix[j])
                .mul(suffix[j + 1])
                .mul(row_product_inv);
        }
    }
}

/// Writes `C⁻¹` (row-major, `k × k`) into `inverse` for the Cauchy matrix on
/// row set `row` and column set `col`.
pub(crate) fn cauchy_inverse_into<F: FieldKernels>(
    row: &[F::Elem],
    col: &[F::Elem],
    inverse: &mut [F::Elem],
    scratch: &mut [F::Elem],
) {
    cauchy_inverse_coefficients_into::<F>(row, col, &[], inverse, &mut [], scratch);
}
