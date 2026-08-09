//! `bits::Ple` against its acceptance. Every dense `Ple` criterion restated
//! over `BitMatrix`, plus the cross-domain differential that is the real
//! point: the same logical matrix carried through `bits::Ple` and through
//! `dense::Ple<Gf8>` one-bit-per-byte yields identical rank, RREF, rank
//! profiles, and kernel bases. Two storage layouts, two inner loops, one
//! answer.

// Index arithmetic across several matrices at once; iterators would obscure it.
#![allow(clippy::needless_range_loop)]

mod common;

use common::draw;
use fgf::Gf8;
use fgf::field::Elem;
use gfm::bits::{Ple, PleScratch, SolveScratch};
use gfm::{BitMatrix, Matrix, Ple as DensePle, PleScratch as DensePleScratch, SolveError};

/// A GF(2) matrix as plain rows of bits.
type Naive = Vec<Vec<bool>>;

/// A matrix of exactly the given rank: `L·U` with `L` unit lower `m × r` and
/// `U` unit upper `r × n`, so the leading `r × r` minors are unit triangular
/// and the product's rank is exactly `r`.
fn naive_with_rank(rows: usize, cols: usize, rank: usize, seed: u64) -> Naive {
    let mut st = seed | 1;
    let l: Naive = (0..rows)
        .map(|i| {
            (0..rank)
                .map(|j| j == i || (j < i && draw(&mut st, 2) == 1))
                .collect()
        })
        .collect();
    let u: Naive = (0..rank)
        .map(|i| {
            (0..cols)
                .map(|j| j == i || (j > i && draw(&mut st, 2) == 1))
                .collect()
        })
        .collect();
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|k| (0..rank).fold(false, |acc, j| acc ^ (l[i][j] && u[j][k])))
                .collect()
        })
        .collect()
}

/// A matrix of uniformly random bits.
fn naive_noise(rows: usize, cols: usize, seed: u64) -> Naive {
    let mut st = seed | 1;
    (0..rows)
        .map(|_| (0..cols).map(|_| draw(&mut st, 2) == 1).collect())
        .collect()
}

/// Packs a naive matrix into `u64` row words.
fn pack(a: &Naive, cols: usize) -> Vec<u64> {
    let words = cols.div_ceil(64);
    let mut out = vec![0u64; a.len() * words];
    for (r, row) in a.iter().enumerate() {
        for (c, &bit) in row.iter().enumerate() {
            if bit {
                out[r * words + c / 64] |= 1u64 << (c % 64);
            }
        }
    }
    out
}

/// Builds a `BitMatrix` from a naive matrix.
fn bit_matrix(a: &Naive, cols: usize) -> BitMatrix {
    BitMatrix::from_rows(a.len(), cols, &pack(a, cols)).unwrap()
}

/// Builds the one-bit-per-byte `Gf8` twin.
fn dense_matrix(a: &Naive, cols: usize) -> Matrix<Gf8> {
    let bytes: Vec<u8> = a
        .iter()
        .flat_map(|row| row.iter().map(|&b| u8::from(b)))
        .collect();
    Matrix::<Gf8>::from_rows(a.len(), cols, &bytes).unwrap()
}

/// Reads a `BitMatrix` back into naive rows.
fn naive_of(m: &BitMatrix) -> Naive {
    (0..m.rows())
        .map(|r| (0..m.cols()).map(|c| m.get(r, c)).collect())
        .collect()
}

/// Naive GF(2) product.
fn naive_mul(a: &Naive, b: &Naive) -> Naive {
    let n = b.first().map_or(0, Vec::len);
    let inner = b.len();
    a.iter()
        .map(|row| {
            (0..n)
                .map(|c| (0..inner).fold(false, |acc, k| acc ^ (row[k] && b[k][c])))
                .collect()
        })
        .collect()
}

/// The `n × n` identity, naive.
fn naive_identity(n: usize) -> Naive {
    (0..n).map(|i| (0..n).map(|j| i == j).collect()).collect()
}

/// A `BitMatrix` of zeros, `rows × cols`.
fn zeros(rows: usize, cols: usize) -> BitMatrix {
    BitMatrix::zeros(rows, cols).unwrap()
}

/// Rank samples covering both ends and the middle.
fn rank_samples(m: usize, n: usize) -> Vec<usize> {
    let full = m.min(n);
    let mut s = vec![0, full];
    if full >= 2 {
        s.push(full / 2);
        s.push(1);
        s.push(full - 1);
    }
    s.sort_unstable();
    s.dedup();
    s
}

/// The public-surface cross-domain check: `bits::Ple` and `dense::Ple<Gf8>`
/// on the same logical matrix agree on rank, both rank profiles, the RREF,
/// and the kernel basis — plus the independent certificates `A·kernel == 0`
/// and `rank + kernel_dim == cols`.
fn check_case(rows: usize, cols: usize, rank: usize, seed: u64) {
    let a = naive_with_rank(rows, cols, rank, seed);
    let bits = Ple::decompose(bit_matrix(&a, cols), &mut PleScratch::new());
    let dense = DensePle::decompose(dense_matrix(&a, cols), &mut DensePleScratch::new());

    assert_eq!(bits.rank(), rank, "constructed rank");
    assert_eq!(bits.rank(), dense.rank(), "rank vs dense");
    assert_eq!(
        bits.row_rank_profile(),
        dense.row_rank_profile(),
        "row rank profile"
    );
    assert_eq!(
        bits.col_rank_profile(),
        dense.col_rank_profile(),
        "column rank profile"
    );

    // RREF agrees bit-for-bit with the dense domain.
    let mut r_bits = zeros(rows, cols);
    bits.rref_into(&mut r_bits);
    let mut r_dense = Matrix::<Gf8>::zeros(rows, cols).unwrap();
    dense.rref_into(&mut r_dense);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(
                r_bits.get(i, j),
                r_dense.get(i, j).is_one(),
                "rref ({i},{j})"
            );
        }
    }

    // Independent of the dense oracle: the RREF is genuinely reduced —
    // exactly `rank` nonzero rows, each with a leading one whose column holds
    // no other one — and it spans the row space of `A` (same rank when the
    // two row sets are stacked).
    let rref = naive_of(&r_bits);
    let nonzero: Vec<&Vec<bool>> = rref.iter().filter(|r| r.iter().any(|&b| b)).collect();
    assert_eq!(
        nonzero.len(),
        rank,
        "rref has the wrong number of pivot rows"
    );
    for row in &nonzero {
        let lead = row.iter().position(|&b| b).unwrap();
        let ones = rref.iter().filter(|r| r[lead]).count();
        assert_eq!(ones, 1, "pivot column {lead} is not reduced");
    }
    let mut stacked = a.clone();
    stacked.extend(rref.iter().cloned());
    let stacked_rank = Ple::decompose(bit_matrix(&stacked, cols), &mut PleScratch::new()).rank();
    assert_eq!(stacked_rank, rank, "rref does not span the row space of A");

    // Kernel: agrees with dense, is a kernel, and has the right dimension.
    let kdim = cols - rank;
    let mut k_bits = zeros(cols, kdim);
    bits.kernel_into(&mut k_bits);
    let mut k_dense = Matrix::<Gf8>::zeros(cols, kdim).unwrap();
    dense.kernel_into(&mut k_dense);
    for i in 0..cols {
        for j in 0..kdim {
            assert_eq!(
                k_bits.get(i, j),
                k_dense.get(i, j).is_one(),
                "kernel ({i},{j})"
            );
        }
    }
    let product = naive_mul(&a, &naive_of(&k_bits));
    for row in &product {
        assert!(row.iter().all(|&b| !b), "A·kernel is not zero");
    }
    // The kernel basis is independent: fed back through `Ple` it has full
    // rank `kdim`.
    if kdim > 0 {
        let basis_rank = Ple::decompose(k_bits.clone(), &mut PleScratch::new()).rank();
        assert_eq!(basis_rank, kdim, "kernel basis is rank-deficient");
    }
}

#[test]
fn cross_domain_small_shapes_every_rank() {
    for m in 0..=16usize {
        for n in 0..=16usize {
            for rank in rank_samples(m, n) {
                check_case(m, n, rank, 0x1B00 ^ ((m << 8 | n | rank << 16) as u64));
            }
        }
    }
}

#[test]
fn cross_domain_rectangular_and_wide() {
    for (m, n) in [(256, 64), (64, 256), (200, 130), (130, 200), (129, 129)] {
        for rank in rank_samples(m, n) {
            check_case(m, n, rank, 0x5C00 ^ ((m << 8 | n | rank << 20) as u64));
        }
    }
}

#[test]
fn cross_domain_word_boundaries() {
    for n in [1, 63, 64, 65, 127, 128, 129] {
        for m in [1, 64, 65] {
            for rank in rank_samples(m, n) {
                check_case(m, n, rank, 0x7A00 ^ ((m << 8 | n | rank << 20) as u64));
            }
        }
    }
}

#[test]
fn rank_is_invariant() {
    // Rank is invariant under row permutation, column permutation, and the
    // addition of one row into another.
    for (m, n) in [(20, 24), (33, 31), (64, 64), (48, 70)] {
        for rank in rank_samples(m, n) {
            let a = naive_with_rank(m, n, rank, 0x9E00 ^ ((m << 8 | n | rank << 16) as u64));
            let base = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new()).rank();
            assert_eq!(base, rank);

            let mut st = 0xC0FFEEu64 ^ (rank as u64);
            // Row swap.
            let mut b = a.clone();
            if m >= 2 {
                let (i, j) = (draw(&mut st, m), draw(&mut st, m));
                b.swap(i, j);
            }
            assert_eq!(
                Ple::decompose(bit_matrix(&b, n), &mut PleScratch::new()).rank(),
                base,
                "row swap changed rank"
            );

            // Column swap.
            let mut c = a.clone();
            if n >= 2 {
                let (i, j) = (draw(&mut st, n), draw(&mut st, n));
                for row in &mut c {
                    row.swap(i, j);
                }
            }
            assert_eq!(
                Ple::decompose(bit_matrix(&c, n), &mut PleScratch::new()).rank(),
                base,
                "column swap changed rank"
            );

            // Add row src into row dst.
            let mut d = a.clone();
            if m >= 2 {
                let dst = draw(&mut st, m);
                let mut src = draw(&mut st, m);
                if src == dst {
                    src = (src + 1) % m;
                }
                for k in 0..n {
                    d[dst][k] ^= d[src][k];
                }
            }
            assert_eq!(
                Ple::decompose(bit_matrix(&d, n), &mut PleScratch::new()).rank(),
                base,
                "row addition changed rank"
            );
        }
    }
}

#[test]
fn det_matches_full_rank() {
    for n in [0, 1, 2, 5, 13, 64, 65] {
        for rank in rank_samples(n, n) {
            let a = naive_with_rank(n, n, rank, 0xDE00 ^ ((n | rank << 16) as u64));
            let bits = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new());
            let dense = DensePle::decompose(dense_matrix(&a, n), &mut DensePleScratch::new());
            assert_eq!(bits.det(), rank == n, "det value at n={n} rank={rank}");
            assert_eq!(
                bits.det(),
                !dense.det().is_zero(),
                "det vs dense at n={n} rank={rank}"
            );
        }
    }
}

#[test]
fn inverse_round_trips() {
    for n in [1, 2, 3, 8, 33, 64, 65, 100] {
        let a = naive_with_rank(n, n, n, 0x11E00 ^ (n as u64));
        let ple = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new());
        let mut inv = zeros(n, n);
        ple.inverse_into(&mut inv).expect("full rank inverts");
        let product = naive_mul(&a, &naive_of(&inv));
        assert_eq!(product, naive_identity(n), "A·A⁻¹ != I at n={n}");
        let product = naive_mul(&naive_of(&inv), &a);
        assert_eq!(product, naive_identity(n), "A⁻¹·A != I at n={n}");
    }
}

#[test]
fn inverse_reports_singular_and_leaves_output_untouched() {
    for (n, rank) in [(4, 2), (8, 7), (16, 0), (33, 30)] {
        let a = naive_with_rank(n, n, rank, 0x5A5A ^ ((n | rank << 8) as u64));
        let ple = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new());
        let mut out = bit_matrix(&naive_noise(n, n, 0xABCD ^ n as u64), n);
        let before = out.clone();
        let err = ple.inverse_into(&mut out).unwrap_err();
        assert_eq!(err, SolveError::Singular { rank, order: n });
        assert_eq!(out, before, "output mutated on Singular");
    }
}

#[test]
fn solve_consistent_systems() {
    for (m, n) in [(10, 10), (17, 12), (12, 17), (64, 64), (80, 48)] {
        for rank in rank_samples(m, n) {
            let a = naive_with_rank(m, n, rank, 0x2C00 ^ ((m << 8 | n | rank << 16) as u64));
            let nrhs = 3;
            let x = naive_noise(n, nrhs, 0x3D00 ^ ((m << 8 | n | rank << 16) as u64));
            let b = naive_mul(&a, &x);
            let amat = bit_matrix(&a, n);
            let ple = Ple::decompose(amat, &mut PleScratch::new());
            let mut out = zeros(n, nrhs);
            ple.solve_into(&bit_matrix(&b, nrhs), &mut out, &mut SolveScratch::new())
                .expect("system is consistent by construction");
            // The returned x' need not equal x when rank-deficient, but must
            // satisfy A·x' == b.
            let checked = naive_mul(&a, &naive_of(&out));
            assert_eq!(checked, b, "A·x' != b at ({m},{n}) rank {rank}");
        }
    }
}

#[test]
fn solve_reports_genuine_inconsistency() {
    // A rank-deficient system whose right-hand side lies outside the column
    // space: force a zero row on the left with a one on the right.
    let (m, n, rank) = (12, 8, 5);
    let a = naive_with_rank(m, n, rank, 0x7799);
    let amat = bit_matrix(&a, n);
    let ple = Ple::decompose(amat, &mut PleScratch::new());
    // Build b as A·x, then flip one entry in a dependent (tail) row so the
    // system becomes inconsistent.
    let x = naive_noise(n, 1, 0x8801);
    let mut b = naive_mul(&a, &x);
    // A tail row of the eliminated system: the last original row is dependent
    // when rank < m. Flip its rhs bit.
    b[m - 1][0] ^= true;
    let mut out = zeros(n, 1);
    let before = out.clone();
    let err = ple
        .solve_into(&bit_matrix(&b, 1), &mut out, &mut SolveScratch::new())
        .unwrap_err();
    let SolveError::Inconsistent { row } = err else {
        panic!("expected Inconsistent, got {err:?}");
    };
    assert!(row >= rank && row < m, "named row {row} is not a tail row");
    assert_eq!(out, before, "output mutated on Inconsistent");
}

#[test]
fn empty_and_zero_matrices() {
    // Degenerate shapes: no panics, rank zero, empty profiles.
    for (m, n) in [(0, 0), (0, 5), (5, 0), (7, 9)] {
        let a = vec![vec![false; n]; m];
        let ple = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new());
        assert_eq!(ple.rank(), 0);
        assert!(ple.row_rank_profile().is_empty());
        assert!(ple.col_rank_profile().is_empty());
        let mut r = zeros(m, n);
        ple.rref_into(&mut r);
        assert_eq!(naive_of(&r), a, "rref of zero matrix is zero");
    }
}

/// The byte-for-byte surface, behind `internals`: `lu`, both permutation
/// actions, rank, both profiles, panel-width independence, and the
/// independent `P·L·U·Q` reassembly certificate.
#[cfg(feature = "internals")]
mod byte_for_byte {
    use super::*;

    /// The naive reassembly `A = P⁻¹·(L·U)·Q⁻¹` from a decomposition's own
    /// `lu`, `p`, and `q` — structurally independent of the elimination.
    fn reassemble(ple: &Ple) -> Naive {
        let (m, n, r) = (ple.rows(), ple.cols(), ple.rank());
        let lu = naive_of(ple.lu());
        let mut out: Naive = (0..m)
            .map(|i| {
                (0..n)
                    .map(|c| {
                        (0..r).fold(false, |acc, t| {
                            let l = if t < i { lu[i][t] } else { t == i };
                            let u = t <= c && lu[t][c];
                            acc ^ (l && u)
                        })
                    })
                    .collect()
            })
            .collect();
        for row in &mut out {
            ple.q().apply_inv(row);
        }
        ple.p().apply_inv(&mut out);
        out
    }

    #[test]
    fn reassembly_certificate() {
        for (m, n) in [(0, 0), (17, 12), (12, 17), (64, 64), (65, 63), (128, 130)] {
            for rank in rank_samples(m, n) {
                let a = naive_with_rank(m, n, rank, 0xBEEF ^ ((m << 8 | n | rank << 16) as u64));
                let ple = Ple::decompose(bit_matrix(&a, n), &mut PleScratch::new());
                assert_eq!(reassemble(&ple), a, "P·L·U·Q != A at ({m},{n}) rank {rank}");
            }
        }
    }

    #[test]
    fn panel_widths_and_dense_agree_byte_for_byte() {
        for (m, n, rank) in [
            (17, 15, 12),
            (16, 16, 16),
            (33, 31, 9),
            (24, 40, 0),
            (40, 24, 24),
            (130, 128, 70),
        ] {
            let a = naive_with_rank(m, n, rank, 0x9A00 ^ ((m << 8 | n | rank << 16) as u64));
            let dense = DensePle::decompose(dense_matrix(&a, n), &mut DensePleScratch::new());
            for width in [1, 2, 3, 7, 64, 256] {
                let bits = Ple::decompose_with_panel_width(
                    bit_matrix(&a, n),
                    &mut PleScratch::new(),
                    width,
                );
                assert_eq!(bits.rank(), dense.rank(), "rank at width {width}");
                assert_eq!(bits.row_rank_profile(), dense.row_rank_profile());
                assert_eq!(bits.col_rank_profile(), dense.col_rank_profile());
                // lu agrees one-bit-per-byte with the dense twin.
                for i in 0..m {
                    for j in 0..n {
                        assert_eq!(
                            bits.lu().get(i, j),
                            dense.lu().get(i, j).is_one(),
                            "lu ({i},{j}) at width {width}"
                        );
                    }
                }
                // Both permutations act identically.
                let mut probe: Vec<usize> = (0..m).collect();
                bits.p().apply(&mut probe);
                let mut dprobe: Vec<usize> = (0..m).collect();
                dense.p().apply(&mut dprobe);
                assert_eq!(probe, dprobe, "row permutation at width {width}");
                let mut probe: Vec<usize> = (0..n).collect();
                bits.q().apply(&mut probe);
                let mut dprobe: Vec<usize> = (0..n).collect();
                dense.q().apply(&mut dprobe);
                assert_eq!(probe, dprobe, "column permutation at width {width}");
            }
        }
    }
    #[test]
    fn m4ri_and_plain_agree_byte_for_byte() {
        for (m, n, rank) in [
            (64, 64, 64),
            (129, 129, 97),
            (200, 130, 111),
            (130, 200, 73),
        ] {
            let a = naive_with_rank(m, n, rank, 0x4A00 ^ ((m << 8 | n | rank << 20) as u64));
            let plain = Ple::decompose_plain(bit_matrix(&a, n), &mut PleScratch::new());
            let table = Ple::decompose_m4ri(bit_matrix(&a, n), &mut PleScratch::new());
            assert_eq!(naive_of(plain.lu()), naive_of(table.lu()));
            assert_eq!(plain.rank(), table.rank());
            assert_eq!(plain.row_rank_profile(), table.row_rank_profile());
            assert_eq!(plain.col_rank_profile(), table.col_rank_profile());
            let mut plain_rows: Vec<_> = (0..m).collect();
            let mut table_rows = plain_rows.clone();
            plain.p().apply(&mut plain_rows);
            table.p().apply(&mut table_rows);
            assert_eq!(plain_rows, table_rows);
            let mut plain_cols: Vec<_> = (0..n).collect();
            let mut table_cols = plain_cols.clone();
            plain.q().apply(&mut plain_cols);
            table.q().apply(&mut table_cols);
            assert_eq!(plain_cols, table_cols);
        }
    }
}

/// The one-elimination invariant for the bit domain: the pivot search
/// `locate_pivot` appears in `bits/ple.rs` and nowhere else in the crate.
#[test]
fn one_bit_pivot_loop_only() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    let mut stack = vec![src];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                if text.contains("locate_pivot") {
                    offenders.push(path);
                }
            }
        }
    }
    assert_eq!(
        offenders,
        [std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("bits")
            .join("ple.rs")],
        "a second bit pivot loop exists"
    );
}
