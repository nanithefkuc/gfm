//! The naive reference implementations the whole suite leans on. Written
//! once, never optimized, and structurally independent of the crate's
//! elimination: one pivot at a time, the full trailing update applied
//! immediately, no panels, no deferred work.
//!
//! Two layers:
//!
//! - the **scalar oracle**, over `Vec<Vec<Elem>>` with per-element field
//!   arithmetic — fully independent of `fgf`'s kernels;
//! - the **packed oracle**, over packed byte rows with the trailing update
//!   delegated to `fgf::ops::mul_add`. Field arithmetic is `fgf`'s own
//!   tested surface; the elimination structure is the independent part.
//!   This layer exists so the big shape×rank sweep is fast enough to run.
//!
//! Nothing here is used by the crate; everything here is a ground truth.
#![allow(dead_code)]
// The loops below index several rows of several matrices by position; the
// index arithmetic is the algorithm, and iterator rewrites obscure it.
#![allow(clippy::needless_range_loop)]

use fgf::field::Elem;
use fgf::{Field, FieldKernels, ops};

use super::common::noise;

/// A matrix as plain rows of elements.
pub type Naive<F> = Vec<Vec<<F as Field>::Elem>>;

/// A matrix as packed byte rows, `cols * F::BYTES` each.
pub type Packed = Vec<Vec<u8>>;

/// Builds a matrix from noise bytes.
pub fn naive_noise<F: FieldKernels>(rows: usize, cols: usize, seed: u64) -> Naive<F> {
    let bytes = noise(rows * cols * F::BYTES, seed);
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let start = (r * cols + c) * F::BYTES;
                    F::read(&bytes[start..start + F::BYTES])
                })
                .collect()
        })
        .collect()
}

/// Packs a naive matrix.
pub fn pack<F: FieldKernels>(a: &Naive<F>) -> Packed {
    a.iter()
        .map(|row| {
            let mut out = vec![0u8; row.len() * F::BYTES];
            for (c, &v) in row.iter().enumerate() {
                F::write(&mut out[c * F::BYTES..(c + 1) * F::BYTES], v);
            }
            out
        })
        .collect()
}

/// Builds a matrix of exactly the given rank: `L·U` with `L` unit lower
/// `m × r` and `U` unit upper `r × n`. The leading `r × r` minor of the
/// product has determinant one, so the rank is exactly `r`.
pub fn naive_with_rank<F: FieldKernels>(
    rows: usize,
    cols: usize,
    rank: usize,
    seed: u64,
) -> Naive<F> {
    assert!(rank <= rows.min(cols));
    if rank == 0 {
        // The product of an m×0 and a 0×n matrix is the m×n zero matrix.
        return (0..rows)
            .map(|_| (0..cols).map(|_| F::Elem::ZERO).collect())
            .collect();
    }
    let mut l = naive_noise::<F>(rows, rank, seed);
    for (i, row) in l.iter_mut().enumerate() {
        for (t, cell) in row.iter_mut().enumerate() {
            *cell = match t.cmp(&i) {
                std::cmp::Ordering::Less => *cell,
                std::cmp::Ordering::Equal => F::Elem::ONE,
                std::cmp::Ordering::Greater => F::Elem::ZERO,
            };
        }
    }
    let mut u = naive_noise::<F>(rank, cols, seed ^ 0x5EED);
    for (t, row) in u.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            if c < t {
                *cell = F::Elem::ZERO;
            } else if c == t {
                *cell = F::Elem::ONE;
            }
        }
    }
    naive_mul::<F>(&l, &u)
}

/// Naive matrix product.
pub fn naive_mul<F: FieldKernels>(a: &Naive<F>, b: &Naive<F>) -> Naive<F> {
    let (m, k) = (a.len(), a.first().map_or(0, Vec::len));
    let n = b.first().map_or(0, Vec::len);
    assert_eq!(k, b.len(), "naive_mul shape mismatch");
    (0..m)
        .map(|i| {
            (0..n)
                .map(|c| {
                    let mut acc = F::Elem::ZERO;
                    for t in 0..k {
                        acc = acc.add(a[i][t].mul(b[t][c]));
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// The `n × n` identity.
pub fn naive_identity<F: FieldKernels>(n: usize) -> Naive<F> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|c| if i == c { F::Elem::ONE } else { F::Elem::ZERO })
                .collect()
        })
        .collect()
}

/// Cofactor-expansion determinant. Exponential; call it at `n <= 6` only.
pub fn naive_det<F: FieldKernels>(a: &Naive<F>) -> F::Elem {
    let n = a.len();
    assert!(a.iter().all(|row| row.len() == n), "det needs square");
    if n == 0 {
        return F::Elem::ONE;
    }
    if n == 1 {
        return a[0][0];
    }
    let mut d = F::Elem::ZERO;
    for c in 0..n {
        let minor: Naive<F> = a[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != c)
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();
        // Characteristic two: the cofactor sign is always one.
        d = d.add(a[0][c].mul(naive_det::<F>(&minor)));
    }
    d
}

/// The result of the scalar oracle elimination, in the crate's `Ple`
/// conventions: `lu` holds the factors below the diagonal and the echelon
/// form on and above it; `p`/`q` are LAPACK-style swap lists (`list[k] = j`
/// means step `k` swapped positions `k` and `j`).
pub struct OraclePle<F: FieldKernels> {
    /// In-place factor storage, `m × n`.
    pub lu: Naive<F>,
    /// Row swap list.
    pub p: Vec<usize>,
    /// Column swap list.
    pub q: Vec<usize>,
    /// The rank found.
    pub rank: usize,
}

/// The scalar reference elimination. Never optimized; used at small shapes
/// where per-element arithmetic is affordable.
pub fn oracle_ple<F: FieldKernels>(a: &Naive<F>) -> OraclePle<F> {
    let (m, n) = (a.len(), a.first().map_or(0, Vec::len));
    let mut lu = a.to_vec();
    let mut p: Vec<usize> = (0..m).collect();
    let mut q: Vec<usize> = (0..n).collect();
    let limit = m.min(n);
    let mut r = 0;
    while r < limit {
        let mut found = None;
        'outer: for j in r..n {
            for i in r..m {
                if !lu[i][j].is_zero() {
                    found = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = found else { break };
        if j != r {
            for row in lu.iter_mut() {
                row.swap(j, r);
            }
            q[r] = j;
        }
        if i != r {
            lu.swap(i, r);
            p[r] = i;
        }
        let pivot_inv = lu[r][r].inv();
        for i2 in (r + 1)..m {
            let entry = lu[i2][r];
            if entry.is_zero() {
                continue;
            }
            let factor = entry.mul(pivot_inv);
            lu[i2][r] = factor;
            for c in (r + 1)..n {
                lu[i2][c] = lu[i2][c].add(factor.mul(lu[r][c]));
            }
        }
        r += 1;
    }
    OraclePle { lu, p, q, rank: r }
}

/// The packed oracle's factorization: byte rows throughout.
pub struct PackedPle {
    /// In-place factor storage, packed `m × n`.
    pub lu: Packed,
    /// Row swap list.
    pub p: Vec<usize>,
    /// Column swap list.
    pub q: Vec<usize>,
    /// The rank found.
    pub rank: usize,
}

impl PackedPle {
    /// Unpacks into naive rows.
    pub fn to_naive<F: FieldKernels>(&self, cols: usize) -> Naive<F> {
        let b = F::BYTES;
        self.lu
            .iter()
            .map(|row| {
                (0..cols)
                    .map(|c| F::read(&row[c * b..(c + 1) * b]))
                    .collect()
            })
            .collect()
    }
}

/// Builds an exact-rank matrix in packed form: `L·U` with `L` unit lower
/// `m × r` and `U` unit upper `r × n`, multiplied out with row kernels.
pub fn packed_with_rank<F: FieldKernels>(
    rows: usize,
    cols: usize,
    rank: usize,
    seed: u64,
) -> Packed {
    assert!(rank <= rows.min(cols));
    let b = F::BYTES;
    if rank == 0 {
        return vec![vec![0u8; cols * b]; rows];
    }
    let mut l: Packed = (0..rows)
        .map(|i| noise(rank * b, seed + i as u64))
        .collect();
    for (i, row) in l.iter_mut().enumerate() {
        for t in 0..rank {
            let v = match t.cmp(&i) {
                std::cmp::Ordering::Less => F::read(&row[t * b..(t + 1) * b]),
                std::cmp::Ordering::Equal => F::Elem::ONE,
                std::cmp::Ordering::Greater => F::Elem::ZERO,
            };
            F::write(&mut row[t * b..(t + 1) * b], v);
        }
    }
    let mut u: Packed = (0..rank)
        .map(|t| noise(cols * b, seed ^ (0x5EED + t as u64)))
        .collect();
    for (t, row) in u.iter_mut().enumerate() {
        for c in 0..=t.min(cols - 1) {
            let v = if c == t { F::Elem::ONE } else { F::Elem::ZERO };
            F::write(&mut row[c * b..(c + 1) * b], v);
        }
    }
    let mut out: Packed = vec![vec![0u8; cols * b]; rows];
    for i in 0..rows {
        for t in 0..rank {
            let f = F::read(&l[i][t * b..(t + 1) * b]);
            if f.is_zero() {
                continue;
            }
            ops::mul_add::<F>(&mut out[i], f, &u[t]);
        }
    }
    out
}

/// The packed reference elimination: identical pivot structure to
/// [`oracle_ple`], with the row update run through `fgf::ops::mul_add`.
pub fn oracle_ple_packed<F: FieldKernels>(a: &Packed, cols: usize) -> PackedPle {
    let m = a.len();
    let n = cols;
    let b = F::BYTES;
    let mut lu: Packed = a.to_vec();
    let mut p: Vec<usize> = (0..m).collect();
    let mut q: Vec<usize> = (0..n).collect();
    let limit = m.min(n);
    let mut r = 0;
    while r < limit {
        let mut found = None;
        'outer: for j in r..n {
            for i in r..m {
                if !F::read(&lu[i][j * b..(j + 1) * b]).is_zero() {
                    found = Some((i, j));
                    break 'outer;
                }
            }
        }
        let Some((i, j)) = found else { break };
        if j != r {
            for row in lu.iter_mut() {
                for w in 0..b {
                    row.swap(j * b + w, r * b + w);
                }
            }
            q[r] = j;
        }
        if i != r {
            lu.swap(i, r);
            p[r] = i;
        }
        let pivot_inv = F::read(&lu[r][r * b..(r + 1) * b]).inv();
        for i2 in (r + 1)..m {
            let entry = F::read(&lu[i2][r * b..(r + 1) * b]);
            if entry.is_zero() {
                continue;
            }
            let factor = entry.mul(pivot_inv);
            F::write(&mut lu[i2][r * b..(r + 1) * b], factor);
            let (head, tail) = lu.split_at_mut(i2);
            let (row_r, row_i) = (&head[r], &mut tail[0]);
            ops::mul_add::<F>(&mut row_i[(r + 1) * b..], factor, &row_r[(r + 1) * b..]);
        }
        r += 1;
    }
    PackedPle { lu, p, q, rank: r }
}

/// Reassembles `P⁻¹·L·U·Q⁻¹` from a packed factorization, with row kernels.
pub fn reassemble_packed<F: FieldKernels>(o: &PackedPle, cols: usize) -> Packed {
    let b = F::BYTES;
    let (m, n, r) = (o.lu.len(), cols, o.rank);
    // U rows: the echelon rows with the below-diagonal factor entries masked.
    let mut u_rows: Packed = o.lu[..r].to_vec();
    for (t, row) in u_rows.iter_mut().enumerate() {
        for cell in row[..t * b].iter_mut() {
            *cell = 0;
        }
    }
    let mut out: Packed = vec![vec![0u8; n * b]; m];
    for i in 0..m {
        for t in 0..r {
            let l = if t < i {
                F::read(&o.lu[i][t * b..(t + 1) * b])
            } else if t == i {
                F::Elem::ONE
            } else {
                F::Elem::ZERO
            };
            if l.is_zero() {
                continue;
            }
            ops::mul_add::<F>(&mut out[i], l, &u_rows[t]);
        }
    }
    // Undo the permutations: replay both swap lists in reverse.
    for k in (0..n).rev() {
        let j = o.q[k];
        if j != k {
            for row in out.iter_mut() {
                for w in 0..b {
                    row.swap(k * b + w, j * b + w);
                }
            }
        }
    }
    for k in (0..m).rev() {
        let i = o.p[k];
        if i != k {
            out.swap(k, i);
        }
    }
    out
}

/// Reassembles `P⁻¹·L·U·Q⁻¹` from an oracle factorization: the independent
/// certificate that the decomposition multiplies back to the input.
pub fn reassemble<F: FieldKernels>(oracle: &OraclePle<F>) -> Naive<F> {
    let m = oracle.lu.len();
    let n = oracle.lu.first().map_or(0, Vec::len);
    let r = oracle.rank;
    let mut out: Naive<F> = (0..m)
        .map(|i| {
            (0..n)
                .map(|c| {
                    let mut acc = F::Elem::ZERO;
                    for t in 0..r {
                        let l = if t < i {
                            oracle.lu[i][t]
                        } else if t == i {
                            F::Elem::ONE
                        } else {
                            F::Elem::ZERO
                        };
                        let u = if t <= c {
                            oracle.lu[t][c]
                        } else {
                            F::Elem::ZERO
                        };
                        acc = acc.add(l.mul(u));
                    }
                    acc
                })
                .collect()
        })
        .collect();
    // Undo the permutations: the swap lists were applied in forward order
    // during elimination, so replay them in reverse.
    for k in (0..n).rev() {
        let j = oracle.q[k];
        if j != k {
            for row in out.iter_mut() {
                row.swap(k, j);
            }
        }
    }
    for k in (0..m).rev() {
        let i = oracle.p[k];
        if i != k {
            out.swap(k, i);
        }
    }
    out
}

/// The RREF of `a`, computed from the scalar oracle's echelon form: undo the
/// column permutation, then normalize each pivot row and clear its pivot
/// column upward.
pub fn oracle_rref<F: FieldKernels>(a: &Naive<F>) -> Naive<F> {
    let o = oracle_ple::<F>(a);
    let (m, n, r) = (o.lu.len(), o.lu.first().map_or(0, Vec::len), o.rank);
    let mut out: Naive<F> = (0..m)
        .map(|i| {
            if i < r {
                // The U echelon row: the stored L multipliers below the
                // diagonal (permuted columns `0..i`) are not part of it.
                let mut row = o.lu[i].clone();
                for cell in row.iter_mut().take(i) {
                    *cell = F::Elem::ZERO;
                }
                row
            } else {
                vec![F::Elem::ZERO; n]
            }
        })
        .collect();
    // Undo Q: replay the column swap list in reverse.
    for k in (0..n).rev() {
        let j = o.q[k];
        if j != k {
            for row in out.iter_mut() {
                row.swap(k, j);
            }
        }
    }
    // Pivot columns in original order: the original index now at position k.
    for k in (0..r).rev() {
        let mut pc = k;
        for (step, &j) in o.q.iter().enumerate().rev() {
            if pc == step {
                pc = j;
            } else if pc == j {
                pc = step;
            }
        }
        let v = out[k][pc];
        if !v.is_one() {
            for cell in out[k].iter_mut() {
                *cell = cell.mul(v.inv());
            }
        }
        for i in 0..k {
            let f = out[i][pc];
            if f.is_zero() {
                continue;
            }
            for c in 0..n {
                out[i][c] = out[i][c].add(f.mul(out[k][c]));
            }
        }
    }
    out
}

/// Applies a swap list to a probe vector `(0..len)` in forward order, the
/// same semantics as `gfm::Perm::apply`.
pub fn apply_swap_list(list: &[usize], len: usize) -> Vec<usize> {
    let mut probe: Vec<usize> = (0..len).collect();
    for (k, &j) in list.iter().enumerate() {
        probe.swap(k, j);
    }
    probe
}

/// The sorted rank profile from a swap list: the first `rank` images,
/// ascending.
pub fn profile_of(list: &[usize], len: usize, rank: usize) -> Vec<usize> {
    let mut out = apply_swap_list(list, len)[..rank].to_vec();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_self_check() {
        // The oracles must agree with each other and with trivial cases:
        // identity decomposes as identity, and the packed and scalar
        // eliminations match on a small random matrix.
        let id = naive_identity::<fgf::Gf8>(5);
        let o = oracle_ple::<fgf::Gf8>(&id);
        assert_eq!(o.rank, 5);
        assert_eq!(reassemble(&o), id);

        let a = naive_noise::<fgf::Gf16>(7, 9, 0x0ACE);
        let scalar = oracle_ple::<fgf::Gf16>(&a);
        let packed = oracle_ple_packed::<fgf::Gf16>(&pack::<fgf::Gf16>(&a), 9);
        assert_eq!(scalar.rank, packed.rank);
        assert_eq!(scalar.lu, packed.to_naive::<fgf::Gf16>(9));
        assert_eq!(scalar.p, packed.p);
        assert_eq!(scalar.q, packed.q);
        assert_eq!(reassemble(&scalar), a);
        assert_eq!(
            reassemble_packed::<fgf::Gf16>(&packed, 9),
            pack::<fgf::Gf16>(&a),
            "packed reassembly certificate"
        );
    }
}
