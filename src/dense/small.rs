#![allow(clippy::needless_range_loop)]

//! Compact square matrices for dense blocks of order at most 64.
//!
//! The const order removes pitch, row-map, and heap-allocation overhead from
//! the small dense blocks produced by inactivation. Field arithmetic remains
//! `fgf`'s; packed right-hand-side row operations use `fgf::ops`.
//! The measured cutoff and three-run comparison with [`Matrix`](crate::Matrix)
//! plus [`Ple`](crate::Ple) are recorded in `BENCHMARKS.md`.

use core::fmt;

use fgf::{FieldKernels, field::Elem, ops};

use crate::SolveError;
use crate::dense::Matrix;

/// A compact, stack-owned `K × K` matrix, intended for `K <= 64`.
#[derive(Clone)]
pub struct SmallMatrix<F: FieldKernels, const K: usize> {
    data: [[F::Elem; K]; K],
    one_byte: [[u8; K]; K],
}

impl<F: FieldKernels, const K: usize> SmallMatrix<F, K> {
    /// A zero matrix.
    ///
    /// # Panics
    ///
    /// Panics if `K > 64`.
    #[must_use]
    pub fn zeros() -> Self {
        assert!(K <= 64, "SmallMatrix order exceeds 64");
        Self {
            data: [[F::Elem::ZERO; K]; K],
            one_byte: [[0; K]; K],
        }
    }

    /// Copies a `K × K` dense matrix into compact storage.
    ///
    /// # Panics
    ///
    /// Panics unless `matrix` is `K × K` or if `K > 64`.
    #[must_use]
    pub fn from_matrix(matrix: &Matrix<F>) -> Self {
        assert_eq!(matrix.rows(), K, "small-matrix row count");
        assert_eq!(matrix.cols(), K, "small-matrix column count");
        let mut result = Self::zeros();
        for row in 0..K {
            for col in 0..K {
                result.set(row, col, matrix.get(row, col));
            }
        }
        result
    }

    /// The element at `(row, col)`.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> F::Elem {
        self.data[row][col]
    }

    /// Sets the element at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, value: F::Elem) {
        self.data[row][col] = value;
        if F::BYTES == 1 {
            F::write(&mut self.one_byte[row][col..=col], value);
        }
    }

    /// Computes the rank without allocating.
    #[must_use]
    pub fn rank(&self) -> usize {
        let mut a = self.data;
        let mut rank = 0;
        while rank < K {
            let mut pivot = None;
            'search: for col in rank..K {
                for row in rank..K {
                    if !a[row][col].is_zero() {
                        pivot = Some((row, col));
                        break 'search;
                    }
                }
            }
            let Some((pivot_row, pivot_col)) = pivot else {
                break;
            };
            a.swap(rank, pivot_row);
            if pivot_col != rank {
                for row in &mut a {
                    row.swap(rank, pivot_col);
                }
            }
            let inverse = a[rank][rank].inv();
            for row in (rank + 1)..K {
                let entry = a[row][rank];
                if entry.is_zero() {
                    continue;
                }
                let factor = entry.mul(inverse);
                a[row][rank] = factor;
                for col in (rank + 1)..K {
                    a[row][col] = a[row][col].add(factor.mul(a[rank][col]));
                }
            }
            rank += 1;
        }
        rank
    }
    fn solve_one_byte(&self, rhs: &Matrix<F>, out: &mut Matrix<F>) -> Result<(), SolveError> {
        let mut a = self.one_byte;
        for row in 0..K {
            out.row_mut(row).copy_from_slice(rhs.row(row));
        }
        for pivot in 0..K {
            let Some(found) = (pivot..K).find(|&row| a[row][pivot] != 0) else {
                return Err(SolveError::Singular {
                    rank: self.rank(),
                    order: K,
                });
            };
            if found != pivot {
                a.swap(found, pivot);
                out.swap_rows(found, pivot);
            }
            let pivot_value = F::read(&a[pivot][pivot..=pivot]);
            let inverse = pivot_value.inv();
            for row in (pivot + 1)..K {
                let entry = F::read(&a[row][pivot..=pivot]);
                if entry.is_zero() {
                    continue;
                }
                let factor = entry.mul(inverse);
                F::write(&mut a[row][pivot..=pivot], factor);
                let prepared = ops::Coeff::<F>::new(factor);
                let (row_dst, row_src) = two_array_rows(&mut a, row, pivot);
                ops::mul_add_with::<F>(&mut row_dst[pivot + 1..], &prepared, &row_src[pivot + 1..]);
                let (rhs_dst, rhs_src) = out.two_live_rows(row, pivot);
                ops::mul_add_with::<F>(rhs_dst, &prepared, rhs_src);
            }
        }
        for pivot in (0..K).rev() {
            for term in (pivot + 1)..K {
                let factor = F::read(&a[pivot][term..=term]);
                if !factor.is_zero() {
                    let (rhs_dst, rhs_src) = out.two_live_rows(pivot, term);
                    crate::row_ops::mul_add::<F>(rhs_dst, factor, rhs_src);
                }
            }
            let pivot_value = F::read(&a[pivot][pivot..=pivot]);
            if !pivot_value.is_one() {
                ops::mul_assign::<F>(out.row_mut(pivot), pivot_value.inv());
            }
        }
        Ok(())
    }

    /// Solves the full-rank square system `self * x = rhs` into `out`.
    ///
    /// The right-hand side and output may have any common column count. Packed
    /// row updates are delegated to `fgf::ops`; the compact coefficient block
    /// never prepares a vector-kernel table.
    ///
    /// # Errors
    ///
    /// [`SolveError::Singular`] if the coefficient matrix is rank-deficient.
    /// `out` may be modified in that case.
    ///
    /// # Panics
    ///
    /// Panics unless `rhs` and `out` both have `K` rows and equal column counts.
    pub fn solve_into(&self, rhs: &Matrix<F>, out: &mut Matrix<F>) -> Result<(), SolveError> {
        assert_eq!(rhs.rows(), K, "right-hand-side row count");
        assert_eq!(out.rows(), K, "solution row count");
        assert_eq!(rhs.cols(), out.cols(), "solution column count");
        if F::BYTES == 1 {
            return self.solve_one_byte(rhs, out);
        }
        let mut a = self.data;
        for row in 0..K {
            out.row_mut(row).copy_from_slice(rhs.row(row));
        }
        for pivot in 0..K {
            let Some(found) = (pivot..K).find(|&row| !a[row][pivot].is_zero()) else {
                return Err(SolveError::Singular {
                    rank: pivot,
                    order: K,
                });
            };
            if found != pivot {
                a.swap(found, pivot);
                out.swap_rows(found, pivot);
            }
            let inverse = a[pivot][pivot].inv();
            if !a[pivot][pivot].is_one() {
                for col in pivot..K {
                    a[pivot][col] = a[pivot][col].mul(inverse);
                }
                ops::mul_assign::<F>(out.row_mut(pivot), inverse);
            }
            for row in 0..K {
                if row == pivot {
                    continue;
                }
                let factor = a[row][pivot];
                if factor.is_zero() {
                    continue;
                }
                a[row][pivot] = F::Elem::ZERO;
                for col in (pivot + 1)..K {
                    a[row][col] = a[row][col].add(factor.mul(a[pivot][col]));
                }
                let (row_dst, row_src) = out.two_live_rows(row, pivot);
                crate::row_ops::mul_add::<F>(row_dst, factor, row_src);
            }
        }
        Ok(())
    }
}

fn two_array_rows<T, const K: usize>(
    rows: &mut [[T; K]; K],
    a: usize,
    b: usize,
) -> (&mut [T; K], &mut [T; K]) {
    debug_assert_ne!(a, b);
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (head, tail) = rows.split_at_mut(hi);
    let row_lo = &mut head[lo];
    let row_hi = &mut tail[0];
    if a < b {
        (row_lo, row_hi)
    } else {
        (row_hi, row_lo)
    }
}

impl<F: FieldKernels, const K: usize> fmt::Debug for SmallMatrix<F, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmallMatrix")
            .field("field", &F::NAME)
            .field("order", &K)
            .finish_non_exhaustive()
    }
}

impl<F: FieldKernels, const K: usize> Default for SmallMatrix<F, K> {
    fn default() -> Self {
        Self::zeros()
    }
}
