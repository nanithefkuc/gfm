//! [`Cauchy`] — the MDS workhorse: `C[i][j] = 1/(x_i + y_j)` over disjoint
//! index sets `X` and `Y`.
//!
//! Every square submatrix of a Cauchy matrix is again Cauchy, hence
//! nonsingular, so a Cauchy generator is MDS by construction and erasure
//! decoding is an `O(k²)` closed-form inverse with no elimination and no
//! pivoting. This type unifies the ecosystem's three index-set policies into
//! one parameterized construction: [`Cauchy::indexed`] for the contiguous
//! `{0..k}` / `{k..k+m}` assignment, [`Cauchy::geometric`] for the
//! Toeplitz-structured geometric-progression assignment, and
//! [`Cauchy::from_points`] for an arbitrary caller-supplied pair of sets.

use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use fgf::FieldKernels;
use fgf::field::Elem;

use crate::GeometryError;
use crate::dense::{Matrix, Ple, PleScratch};
use crate::structured::cauchy_inv::{cauchy_inverse_into, cauchy_scratch_len};

/// A Cauchy matrix over field `F`, defined by disjoint index sets `X`
/// (`row_vars`, length `k`) and `Y` (`col_vars`, length `m`).
///
/// The matrix is `k × m` with `C[i][j] = (x_i + y_j)⁻¹`. It is not
/// materialized unless asked; coefficients are recomputed on access.
#[derive(Clone)]
pub struct Cauchy<F: FieldKernels> {
    row: Vec<F::Elem>,
    col: Vec<F::Elem>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Cauchy<F> {
    /// A Cauchy matrix from explicit index sets.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Capacity`] if the two sets together need more distinct
    /// points than the field holds; [`GeometryError::Collision`] if any point
    /// repeats within a set or across the two (which would make `x_i + y_j`
    /// zero, or a submatrix singular).
    pub fn from_points(row: &[F::Elem], col: &[F::Elem]) -> Result<Self, GeometryError> {
        let requested = row.len() + col.len();
        if (requested as u128) > F::ORDER {
            return Err(GeometryError::Capacity {
                requested,
                order: F::ORDER,
            });
        }
        // Distinct within each set and disjoint across them: the union is a set
        // of distinct points. Quadratic, but construction is not a hot path and
        // the sets are small.
        let all = || row.iter().chain(col.iter());
        for (i, a) in all().enumerate() {
            for b in all().skip(i + 1) {
                if a == b {
                    return Err(GeometryError::Collision {
                        value: elem_to_u64::<F>(*a),
                    });
                }
            }
        }
        Ok(Self {
            row: row.to_vec(),
            col: col.to_vec(),
            field: PhantomData,
        })
    }

    /// The contiguous assignment `X = {0, …, k−1}`, `Y = {k, …, k+m−1}`,
    /// interpreting each index as the field element with that little-endian
    /// byte value.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Capacity`] if `k + m` exceeds the field order.
    pub fn indexed(k: usize, m: usize) -> Result<Self, GeometryError> {
        let requested = k + m;
        if (requested as u128) > F::ORDER {
            return Err(GeometryError::Capacity {
                requested,
                order: F::ORDER,
            });
        }
        let row: Vec<F::Elem> = (0..k).map(elem_from_index::<F>).collect();
        let col: Vec<F::Elem> = (k..k + m).map(elem_from_index::<F>).collect();
        Self::from_points(&row, &col)
    }

    /// The geometric assignment `X = {g⁰, …, g^{k−1}}`,
    /// `Y = {g^k, …, g^{k+m−1}}` for a generator `g`, which makes the matrix
    /// Toeplitz-structured. Passing `F::GENERATOR` gives the field's canonical
    /// multiplicative generator.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Capacity`] if `k + m` exceeds the field order, or
    /// [`GeometryError::Collision`] if `g`'s multiplicative order is too small
    /// for the powers to stay distinct.
    pub fn geometric(k: usize, m: usize, g: F::Elem) -> Result<Self, GeometryError> {
        let requested = k + m;
        if (requested as u128) > F::ORDER {
            return Err(GeometryError::Capacity {
                requested,
                order: F::ORDER,
            });
        }
        let row: Vec<F::Elem> = (0..k as u64).map(|e| g.pow(e)).collect();
        let col: Vec<F::Elem> = (k as u64..(k + m) as u64).map(|e| g.pow(e)).collect();
        Self::from_points(&row, &col)
    }

    /// Rows (`|X|`).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.row.len()
    }

    /// Columns (`|Y|`).
    #[must_use]
    pub fn cols(&self) -> usize {
        self.col.len()
    }

    /// The row index set `X`.
    #[must_use]
    pub fn row_vars(&self) -> &[F::Elem] {
        &self.row
    }

    /// The column index set `Y`.
    #[must_use]
    pub fn col_vars(&self) -> &[F::Elem] {
        &self.col
    }

    /// The `(i, j)` coefficient `(x_i + y_j)⁻¹`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn coeff(&self, i: usize, j: usize) -> F::Elem {
        self.row[i].add(self.col[j]).inv()
    }

    /// Writes the full `k × m` matrix into `out`.
    ///
    /// # Panics
    ///
    /// Panics unless `out` is `rows × cols`.
    pub fn materialize_into(&self, out: &mut Matrix<F>) {
        assert_eq!(
            (out.rows(), out.cols()),
            (self.rows(), self.cols()),
            "cauchy output shape mismatch"
        );
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                out.set(i, j, self.coeff(i, j));
            }
        }
    }

    /// Writes `C⁻¹` into `out` via the closed form, `O(k²)`.
    ///
    /// # Panics
    ///
    /// Panics unless the matrix is square and `out` is `k × k`.
    pub fn inverse_into(&self, out: &mut Matrix<F>) {
        let k = self.rows();
        assert_eq!(self.cols(), k, "cauchy inverse requires a square matrix");
        assert_eq!(
            (out.rows(), out.cols()),
            (k, k),
            "cauchy inverse output shape mismatch"
        );
        let mut inverse = vec![F::Elem::ZERO; k * k];
        let mut scratch = vec![F::Elem::ZERO; cauchy_scratch_len(k)];
        cauchy_inverse_into::<F>(&self.row, &self.col, &mut inverse, &mut scratch);
        for r in 0..k {
            for c in 0..k {
                out.set(r, c, inverse[r * k + c]);
            }
        }
    }

    /// Whether the systematic generator `[I_k | C]` is MDS: every square
    /// submatrix of `C` is nonsingular.
    ///
    /// Exhaustive minor enumeration — exponential in `min(rows, cols)`, a
    /// self-check for small sizes, never a hot path.
    #[must_use]
    pub fn is_mds(&self) -> bool {
        let (k, m) = (self.rows(), self.cols());
        let r_max = k.min(m);
        let mut scratch = PleScratch::new();
        for r in 1..=r_max {
            for row_sel in combinations(k, r) {
                for col_sel in combinations(m, r) {
                    let mut minor = Matrix::<F>::zeros(r, r)
                        .unwrap_or_else(|_| unreachable!("r is small and bounded"));
                    for (a, &ri) in row_sel.iter().enumerate() {
                        for (b, &cj) in col_sel.iter().enumerate() {
                            minor.set(a, b, self.coeff(ri, cj));
                        }
                    }
                    if Ple::decompose(minor, &mut scratch).rank() < r {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl<F: FieldKernels> core::fmt::Debug for Cauchy<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Cauchy")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .finish_non_exhaustive()
    }
}

/// The field element whose little-endian byte encoding is `i`.
fn elem_from_index<F: FieldKernels>(i: usize) -> F::Elem {
    let bytes = (i as u64).to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

/// The little-endian byte encoding of `e` as a `u64` (fields are ≤ 8 bytes).
fn elem_to_u64<F: FieldKernels>(e: F::Elem) -> u64 {
    let mut bytes = [0u8; 8];
    F::write(&mut bytes[..F::BYTES], e);
    u64::from_le_bytes(bytes)
}

/// Every `r`-element subset of `{0, …, n−1}`, in ascending lexicographic
/// order. Used only by [`Cauchy::is_mds`], at small sizes.
fn combinations(n: usize, r: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if r > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        out.push(idx.clone());
        // Advance to the next combination.
        let mut i = r;
        while i > 0 {
            i -= 1;
            if idx[i] != i + n - r {
                idx[i] += 1;
                for j in i + 1..r {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                return out;
            }
        }
        if r == 0 {
            return out;
        }
    }
}
