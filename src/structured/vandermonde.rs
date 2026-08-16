//! [`Vandermonde`] — `V[i][j] = x_i^j` over distinct evaluation points, with
//! `O(n²)` inversion by Lagrange interpolation.
//!
//! The rows of `V⁻¹` are the coefficient vectors of the Lagrange basis
//! polynomials: `V⁻¹[j][i]` is the coefficient of `x^j` in
//! `L_i(x) = ∏_{m≠i}(x − x_m) / ∏_{m≠i}(x_i − x_m)`, obtained by dividing the
//! master polynomial `∏_m(x − x_m)` by each `(x − x_i)` and normalizing. In
//! characteristic two subtraction is addition, so `(x − x_m)` is `(x + x_m)`.
//!
//! # Warning
//!
//! A full Vandermonde matrix over distinct points is nonsingular, but a
//! *submatrix* of one is **not necessarily** nonsingular. A "Vandermonde
//! Reed–Solomon" erasure code — a systematic `[I | V]` generator — is
//! therefore **not MDS for every erasure pattern**: some sets of `k` received
//! symbols select a singular `k × k` submatrix and cannot be decoded. This has
//! shipped as a data-loss bug in real storage systems. Use [`super::Cauchy`]
//! when an MDS guarantee is required; a concrete singular submatrix is
//! exhibited by the `singular_submatrix_exists` test in `tests/structured.rs`.

use core::marker::PhantomData;

use alloc::vec::Vec;

use fgf::FieldKernels;
use fgf::field::Elem;

use univariate::Polynomial;

use crate::GeometryError;
use crate::dense::Matrix;

/// An `n × n` Vandermonde matrix over field `F`, defined by `n` distinct
/// evaluation points; `V[i][j] = x_i^j`.
#[derive(Clone)]
pub struct Vandermonde<F: FieldKernels> {
    points: Vec<F::Elem>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Vandermonde<F> {
    /// A Vandermonde matrix from explicit, distinct evaluation points.
    ///
    /// # Errors
    ///
    /// [`GeometryError::Capacity`] if more points are requested than the field
    /// holds; [`GeometryError::Collision`] if any point repeats.
    pub fn from_points(points: &[F::Elem]) -> Result<Self, GeometryError> {
        if (points.len() as u128) > F::ORDER {
            return Err(GeometryError::Capacity {
                requested: points.len(),
                order: F::ORDER,
            });
        }
        for (i, a) in points.iter().enumerate() {
            for b in points.iter().skip(i + 1) {
                if a == b {
                    let mut bytes = [0u8; 8];
                    F::write(&mut bytes[..F::BYTES], *a);
                    return Err(GeometryError::Collision {
                        value: u64::from_le_bytes(bytes),
                    });
                }
            }
        }
        Ok(Self {
            points: points.to_vec(),
            field: PhantomData,
        })
    }

    /// The order `n`.
    #[must_use]
    pub fn order(&self) -> usize {
        self.points.len()
    }

    /// The evaluation points.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// The `(i, j)` coefficient `x_i^j`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn coeff(&self, i: usize, j: usize) -> F::Elem {
        assert!(
            i < self.points.len() && j < self.points.len(),
            "index out of bounds"
        );
        self.points[i].pow(j as u64)
    }

    /// Writes the full `n × n` matrix into `out`.
    ///
    /// # Panics
    ///
    /// Panics unless `out` is `n × n`.
    pub fn materialize_into(&self, out: &mut Matrix<F>) {
        let n = self.points.len();
        assert_eq!(
            (out.rows(), out.cols()),
            (n, n),
            "vandermonde output shape mismatch"
        );
        for i in 0..n {
            let mut power = F::Elem::ONE;
            for j in 0..n {
                out.set(i, j, power);
                power = power.mul(self.points[i]);
            }
        }
    }

    /// Writes `V⁻¹` into `out` via Lagrange interpolation, `O(n²)`.
    ///
    /// # Panics
    ///
    /// Panics unless `out` is `n × n`.
    pub fn inverse_into(&self, out: &mut Matrix<F>) {
        let n = self.points.len();
        assert_eq!(
            (out.rows(), out.cols()),
            (n, n),
            "vandermonde inverse output shape mismatch"
        );
        if n == 0 {
            return;
        }
        // Master polynomial P(x) = ∏_m (x + x_m), degree n, monic, built
        // through the shared univariate ring.
        let mut master = Polynomial::<F>::one().expect("one is a constant");
        for &xm in &self.points {
            master = master.multiply_x_plus(xm).expect("master product");
        }
        // For each point, N_i(x) = P(x) / (x + x_i) and the normalizer
        // d_i = N_i(x_i) = ∏_{m≠i}(x_i + x_m).
        for (i, &xi) in self.points.iter().enumerate() {
            let linear =
                Polynomial::<F>::from_coefficients(&[xi, F::Elem::ONE]).expect("linear factor");
            let (quotient, remainder) = master.div_rem(&linear).expect("exact linear factor");
            debug_assert!(remainder.is_zero());
            let inv_di = quotient.evaluate(xi).inv();
            for j in 0..n {
                out.set(j, i, quotient.coefficient(j).mul(inv_di));
            }
        }
    }
}

impl<F: FieldKernels> core::fmt::Debug for Vandermonde<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vandermonde")
            .field("order", &self.order())
            .finish_non_exhaustive()
    }
}
