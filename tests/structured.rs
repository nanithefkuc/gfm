//! Structured matrices against their acceptance: batch inversion versus
//! elementwise `inv` on every field (zero included), Cauchy round-trips and
//! the closed form against `Ple::inverse` under both index-set policies,
//! `is_mds` by exhaustive minors, and the Vandermonde singular-submatrix trap.

#![allow(clippy::needless_range_loop)]

mod common;

use common::noise;
use fgf::field::{Elem, Field};
use fgf::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, FieldKernels, Gf8, Gf16, Gf32, Gf64};
use gfm::{
    Cauchy, GeometryError, Matrix, Ple, PleScratch, Vandermonde, batch_invert,
    cauchy_inverse_coefficients_into, cauchy_scratch_len,
};

/// The field element whose little-endian encoding is `i`.
fn elem<F: FieldKernels>(i: u64) -> F::Elem {
    let bytes = i.to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

/// `n` elements read from deterministic noise, with a few forced to zero so
/// the zero path is exercised.
fn elems<F: FieldKernels>(n: usize, seed: u64) -> Vec<F::Elem> {
    let bytes = noise(n * F::BYTES, seed);
    let mut v: Vec<F::Elem> = (0..n)
        .map(|i| F::read(&bytes[i * F::BYTES..(i + 1) * F::BYTES]))
        .collect();
    // Force a scatter of zeros.
    for i in (0..n).step_by(5) {
        v[i] = F::Elem::ZERO;
    }
    v
}

/// Naive matrix product over `F`, returned as rows of elements.
fn naive_mul<F: FieldKernels>(a: &Matrix<F>, b: &Matrix<F>) -> Vec<Vec<F::Elem>> {
    (0..a.rows())
        .map(|i| {
            (0..b.cols())
                .map(|j| {
                    let mut acc = F::Elem::ZERO;
                    for t in 0..a.cols() {
                        acc = acc.add(a.get(i, t).mul(b.get(t, j)));
                    }
                    acc
                })
                .collect()
        })
        .collect()
}

/// Asserts `m` is the `n × n` identity.
fn assert_identity<F: FieldKernels>(m: &[Vec<F::Elem>], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { F::Elem::ONE } else { F::Elem::ZERO };
            assert_eq!(m[i][j], want, "identity mismatch at ({i},{j})");
        }
    }
}

fn check_batch_invert<F: FieldKernels>(seed: u64) {
    for n in [0usize, 1, 2, 7, 16, 33, 64] {
        let values = elems::<F>(n, seed ^ (n as u64));
        let mut got = values.clone();
        batch_invert::<F>(&mut got);
        for (i, (&v, &g)) in values.iter().zip(&got).enumerate() {
            assert_eq!(g, v.inv(), "batch_invert[{i}] disagrees with inv() (n={n})");
        }
    }
}

#[test]
fn batch_invert_matches_elementwise_every_field() {
    check_batch_invert::<Gf8>(0x1000);
    check_batch_invert::<Gf16>(0x2000);
    check_batch_invert::<Gf32>(0x3000);
    check_batch_invert::<Gf64>(0x4000);
    check_batch_invert::<FanPaar8>(0x5000);
    check_batch_invert::<FanPaar16>(0x6000);
    check_batch_invert::<FanPaar32>(0x7000);
    check_batch_invert::<FanPaar64>(0x8000);
}

#[test]
fn batch_invert_handles_the_zero_element() {
    // inv(0) == 0 (I3): a zero in the buffer must not collapse the product.
    let mut v = [
        Gf8::read(&[0]),
        Gf8::read(&[5]),
        Gf8::read(&[0]),
        Gf8::read(&[9]),
    ];
    let expect: Vec<_> = v.iter().map(|e| e.inv()).collect();
    batch_invert::<Gf8>(&mut v);
    assert_eq!(v.to_vec(), expect);
    assert_eq!(v[0], <Gf8 as fgf::Field>::Elem::ZERO);
}

/// Materializes a square Cauchy, checks `C·C⁻¹ == I` by independent multiply,
/// and checks the closed form agrees entry-for-entry with `Ple::inverse`.
fn check_cauchy_square<F: FieldKernels>(cauchy: &Cauchy<F>) {
    let k = cauchy.rows();
    assert_eq!(cauchy.cols(), k);
    let mut c = Matrix::<F>::zeros(k, k).unwrap();
    cauchy.materialize_into(&mut c);

    let mut cinv = Matrix::<F>::zeros(k, k).unwrap();
    cauchy.inverse_into(&mut cinv);
    assert_identity::<F>(&naive_mul(&c, &cinv), k);
    assert_identity::<F>(&naive_mul(&cinv, &c), k);

    // Closed form vs the general elimination inverse, entry for entry.
    let ple = Ple::decompose(c.clone(), &mut PleScratch::new());
    let mut ple_inv = Matrix::<F>::zeros(k, k).unwrap();
    ple.inverse_into(&mut ple_inv)
        .expect("cauchy is nonsingular");
    for i in 0..k {
        for j in 0..k {
            assert_eq!(
                cinv.get(i, j),
                ple_inv.get(i, j),
                "closed form vs Ple::inverse at ({i},{j}), k={k}"
            );
        }
    }
}

#[test]
fn cauchy_inverse_indexed_policy() {
    for k in 1..=64usize {
        let cauchy = Cauchy::<Gf8>::indexed(k, k).expect("k+k <= 256");
        check_cauchy_square(&cauchy);
    }
}

#[test]
fn cauchy_inverse_geometric_policy() {
    let g = <Gf8 as fgf::Field>::GENERATOR;
    for k in 1..=64usize {
        let cauchy = Cauchy::<Gf8>::geometric(k, k, g).expect("generator has full order");
        check_cauchy_square(&cauchy);
    }
}

#[test]
fn cauchy_inverse_wider_fields() {
    for k in [1usize, 2, 8, 32, 64] {
        check_cauchy_square(&Cauchy::<Gf16>::indexed(k, k).unwrap());
        check_cauchy_square(&Cauchy::<Gf32>::indexed(k, k).unwrap());
    }
}

#[test]
fn cauchy_from_points_arbitrary() {
    // The pool policy: arbitrary disjoint sets. Use a scattered selection.
    let row: Vec<_> = [1u64, 7, 42, 200].iter().map(|&i| elem::<Gf8>(i)).collect();
    let col: Vec<_> = [3u64, 9, 100, 255]
        .iter()
        .map(|&i| elem::<Gf8>(i))
        .collect();
    let cauchy = Cauchy::<Gf8>::from_points(&row, &col).unwrap();
    check_cauchy_square(&cauchy);
}

#[test]
fn cauchy_fused_extra_coefficients_match_multiply() {
    let row: Vec<_> = [1u64, 7, 42].iter().map(|&i| elem::<Gf8>(i)).collect();
    let col: Vec<_> = [3u64, 9, 100].iter().map(|&i| elem::<Gf8>(i)).collect();
    let extra: Vec<_> = [17u64, 33].iter().map(|&i| elem::<Gf8>(i)).collect();
    let k = row.len();
    let mut inverse = vec![<Gf8 as Field>::Elem::ZERO; k * k];
    let mut fused = vec![<Gf8 as Field>::Elem::ZERO; extra.len() * k];
    let mut scratch = vec![<Gf8 as Field>::Elem::ZERO; cauchy_scratch_len(k)];
    cauchy_inverse_coefficients_into::<Gf8>(
        &row,
        &col,
        &extra,
        &mut inverse,
        &mut fused,
        &mut scratch,
    );

    for (z_pos, &z) in extra.iter().enumerate() {
        for inverse_row in 0..k {
            let expected = (0..k).fold(<Gf8 as Field>::Elem::ZERO, |acc, i| {
                acc.add(inverse[inverse_row * k + i].mul(row[i].add(z).inv()))
            });
            assert_eq!(fused[z_pos * k + inverse_row], expected);
        }
    }
}

#[test]
fn cauchy_is_mds_matches_reference() {
    // Every valid Cauchy is MDS; srs's is_mds returns true across these sizes.
    for k in 1..=5usize {
        for m in 1..=5usize {
            assert!(
                Cauchy::<Gf8>::indexed(k, m).unwrap().is_mds(),
                "indexed Cauchy {k}x{m} should be MDS"
            );
        }
    }
    // The geometric policy is MDS too.
    let g = <Gf8 as fgf::Field>::GENERATOR;
    assert!(Cauchy::<Gf8>::geometric(4, 4, g).unwrap().is_mds());
}

#[test]
fn cauchy_rejects_bad_construction() {
    // Capacity: too many points for the field.
    assert_eq!(
        Cauchy::<Gf8>::indexed(200, 200).unwrap_err(),
        GeometryError::Capacity {
            requested: 400,
            order: 256
        }
    );
    // Collision: overlapping sets.
    let row = [elem::<Gf8>(1), elem::<Gf8>(2)];
    let col = [elem::<Gf8>(2), elem::<Gf8>(3)];
    assert!(matches!(
        Cauchy::<Gf8>::from_points(&row, &col),
        Err(GeometryError::Collision { .. })
    ));
}

#[test]
fn vandermonde_round_trips() {
    for n in [1usize, 2, 5, 16, 40] {
        let points: Vec<_> = (0..n as u64).map(elem::<Gf8>).collect();
        let v = Vandermonde::<Gf8>::from_points(&points).unwrap();
        let mut vm = Matrix::<Gf8>::zeros(n, n).unwrap();
        v.materialize_into(&mut vm);
        let mut vinv = Matrix::<Gf8>::zeros(n, n).unwrap();
        v.inverse_into(&mut vinv);
        assert_identity::<Gf8>(&naive_mul(&vm, &vinv), n);
        assert_identity::<Gf8>(&naive_mul(&vinv, &vm), n);
    }
}

#[test]
fn vandermonde_inverse_matches_ple() {
    for n in [3usize, 7, 20] {
        let points: Vec<_> = (0..n as u64).map(elem::<Gf16>).collect();
        let v = Vandermonde::<Gf16>::from_points(&points).unwrap();
        let mut vm = Matrix::<Gf16>::zeros(n, n).unwrap();
        v.materialize_into(&mut vm);
        let mut vinv = Matrix::<Gf16>::zeros(n, n).unwrap();
        v.inverse_into(&mut vinv);
        let ple = Ple::decompose(vm.clone(), &mut PleScratch::new());
        let mut ple_inv = Matrix::<Gf16>::zeros(n, n).unwrap();
        ple.inverse_into(&mut ple_inv).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(vinv.get(i, j), ple_inv.get(i, j), "({i},{j}) n={n}");
            }
        }
    }
}

/// The documented trap: a *submatrix* of a Vandermonde matrix need not be
/// nonsingular, so "Vandermonde RS" is not MDS for every erasure pattern.
/// This exhibits a concrete singular square submatrix, backing the `# Warning`
/// on [`Vandermonde`].
#[test]
fn singular_submatrix_exists() {
    let n = 7usize;
    let points: Vec<_> = (0..n as u64).map(elem::<Gf8>).collect();
    let v = Vandermonde::<Gf8>::from_points(&points).unwrap();
    // Search every square submatrix (row subset × column subset) for one that
    // is rank-deficient.
    let mut found = None;
    'search: for r in 2..=n {
        for rows in combinations(n, r) {
            for cols in combinations(n, r) {
                let mut minor = Matrix::<Gf8>::zeros(r, r).unwrap();
                for (a, &ri) in rows.iter().enumerate() {
                    for (b, &cj) in cols.iter().enumerate() {
                        minor.set(a, b, v.coeff(ri, cj));
                    }
                }
                if Ple::decompose(minor, &mut PleScratch::new()).rank() < r {
                    found = Some((rows.clone(), cols.clone()));
                    break 'search;
                }
            }
        }
    }
    assert!(
        found.is_some(),
        "expected a singular Vandermonde submatrix to exist"
    );
}

/// Every `r`-subset of `0..n`, ascending.
fn combinations(n: usize, r: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if r > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..r).collect();
    loop {
        out.push(idx.clone());
        let mut i = r;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - r {
                idx[i] += 1;
                for j in i + 1..r {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
}
