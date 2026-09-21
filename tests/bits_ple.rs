//! `bits::Ple` against its acceptance. Every dense `Ple` criterion restated
//! over `BitMatrix`, plus the cross-domain differential that is the real
//! point: the same logical matrix carried through `bits::Ple` and through
//! `dense::Ple<Gf8B>` one-bit-per-byte yields identical rank, RREF, rank
//! profiles, and kernel bases. Narrow panel windows, matrices whose leading
//! columns are dead, storage reclaimed from a finished decomposition, and
//! default scratch answer here too. Two storage layouts, two inner loops,
//! one answer.

// Index arithmetic across several matrices at once; iterators would obscure it.
#![allow(clippy::needless_range_loop)]

mod common;

use common::{draw, noise};
use fgf::Gf8B;
use fgf::field::Elem;
use gfm::bits::{Ple, PleScratch, SolveScratch};
use gfm::internals::{BitPleInternals, PleInternals};
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

/// A rank-`rank` matrix with `dead` zero columns prepended: column pivoting
/// is mandatory, never incidental.
fn with_dead_leading(rows: usize, cols: usize, dead: usize, rank: usize, seed: u64) -> Naive {
    naive_with_rank(rows, cols - dead, rank, seed)
        .into_iter()
        .map(|row| {
            let mut out = vec![false; dead];
            out.extend(row);
            out
        })
        .collect()
}

/// Packs a naive matrix into `fgf::bits` row bytes.
fn pack(a: &Naive, cols: usize) -> Vec<u8> {
    let live = cols.div_ceil(8);
    let mut out = vec![0u8; a.len() * live];
    for (r, row) in a.iter().enumerate() {
        for (c, &bit) in row.iter().enumerate() {
            if bit {
                out[r * live + c / 8] |= 1 << (c % 8);
            }
        }
    }
    out
}

/// Builds a `BitMatrix` from a naive matrix.
fn bit_matrix(a: &Naive, cols: usize) -> BitMatrix {
    BitMatrix::from_rows(a.len(), cols, &pack(a, cols)).unwrap()
}

/// Builds the one-bit-per-byte `Gf8B` twin.
fn dense_matrix(a: &Naive, cols: usize) -> Matrix<Gf8B> {
    let bytes: Vec<u8> = a
        .iter()
        .flat_map(|row| row.iter().map(|&b| u8::from(b)))
        .collect();
    Matrix::<Gf8B>::from_rows(a.len(), cols, &bytes).unwrap()
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

/// The public-surface cross-domain check: `bits::Ple` and `dense::Ple<Gf8B>`
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
    let mut r_dense = Matrix::<Gf8B>::zeros(rows, cols).unwrap();
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
    let mut k_dense = Matrix::<Gf8B>::zeros(cols, kdim).unwrap();
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
fn rref_skips_dead_leading_columns() {
    for &(rows, cols, dead) in &[(6, 9, 1), (9, 6, 2), (8, 8, 3)] {
        let full = rows.min(cols - dead);
        for rank in [0, 1, full / 2, full] {
            let seed = 0xB17 ^ ((rows << 12 | cols << 8 | dead << 4 | rank) as u64);
            let a = with_dead_leading(rows, cols, dead, rank, seed);
            let bits = Ple::decompose(bit_matrix(&a, cols), &mut PleScratch::new());
            assert_eq!(bits.rank(), rank);
            assert!(
                bits.col_rank_profile().iter().all(|&c| c >= dead),
                "profile reaches into dead columns"
            );
            // The RREF is genuinely reduced: one leading one per nonzero
            // row, zeros elsewhere in its column.
            let mut r_bits = zeros(rows, cols);
            bits.rref_into(&mut r_bits);
            let nonzero = (0..rows)
                .filter(|&r| (0..cols).any(|c| r_bits.get(r, c)))
                .count();
            assert_eq!(nonzero, rank, "wrong number of pivot rows");
            for r in 0..rows {
                if let Some(lead) = (0..cols).find(|&c| r_bits.get(r, c)) {
                    let ones = (0..rows).filter(|&rr| r_bits.get(rr, lead)).count();
                    assert_eq!(ones, 1, "pivot column {lead} is not reduced");
                }
            }
            // And it agrees with the dense twin bit for bit.
            let dense = DensePle::decompose(dense_matrix(&a, cols), &mut DensePleScratch::new());
            let mut r_dense = Matrix::<Gf8B>::zeros(rows, cols).unwrap();
            dense.rref_into(&mut r_dense);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        r_bits.get(r, c),
                        r_dense.get(r, c).is_one(),
                        "rref ({r}, {c})"
                    );
                }
            }
        }
    }
}

#[test]
fn ple_moves_columns_when_the_pivot_is_not_leading() {
    // The anti-diagonal forces column exchanges, and RREF is the identity.
    let n = 5;
    let mut data = vec![0u8; n];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte |= 1 << (n - 1 - i);
    }
    let a = BitMatrix::from_rows(n, n, &data).unwrap();
    let ple = Ple::decompose(a, &mut PleScratch::new());
    assert_eq!(ple.rank(), n);
    assert!(ple.det());
    let mut rref = zeros(n, n);
    ple.rref_into(&mut rref);
    assert_eq!(rref, BitMatrix::identity(n).unwrap());
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
fn permutation_matrix_inverts_to_its_transpose() {
    // A reversal permutation: `U` has zeros above the diagonal wherever the
    // permutation displaces, and unit pivots throughout.
    let n = 6;
    let mut data = vec![0u8; n];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte |= 1 << (n - 1 - i);
    }
    let a = BitMatrix::from_rows(n, n, &data).unwrap();
    let ple = Ple::decompose(a, &mut PleScratch::new());
    assert_eq!(ple.rank(), n);
    let mut inv = zeros(n, n);
    ple.inverse_into(&mut inv).unwrap();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(inv.get(i, j), i + j == n - 1, "inverse ({i}, {j})");
        }
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
fn default_scratch_drives_a_decomposition_and_a_solve() {
    // `default()` scratch is not a placeholder: it carries a real
    // decomposition, its determinant, and a real solve.
    let mut b = BitMatrix::identity(3).unwrap();
    b.set(0, 1, true);
    let bit_ple = Ple::decompose(b, &mut PleScratch::default());
    assert_eq!(bit_ple.rank(), 3);
    let det = bit_ple.det();
    assert!(det);
    let mut bit_out = zeros(3, 1);
    let mut bit_rhs = zeros(3, 1);
    bit_rhs.set(2, 0, true);
    bit_ple
        .solve_into(&bit_rhs, &mut bit_out, &mut SolveScratch::default())
        .unwrap();
    assert!(bit_out.get(2, 0));
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

#[test]
fn into_matrix_reclaims_usable_storage() {
    for (rows, cols) in [(7usize, 9usize), (16, 16), (65, 33)] {
        let live = cols.div_ceil(8);
        let mut data = noise(rows * live, 0xB17 ^ ((rows << 8 | cols) as u64));
        for chunk in data.chunks_exact_mut(live) {
            if !cols.is_multiple_of(8) {
                chunk[live - 1] &= (1 << (cols % 8)) - 1;
            }
        }
        let a = BitMatrix::from_rows(rows, cols, &data).unwrap();
        let ple = Ple::decompose(a, &mut PleScratch::new());
        let mut reclaimed = ple.into_matrix();
        assert_eq!((reclaimed.rows(), reclaimed.cols()), (rows, cols));
        let mut refill = noise(rows * live, 0xB18 ^ ((rows << 8 | cols) as u64));
        for chunk in refill.chunks_exact_mut(live) {
            if !cols.is_multiple_of(8) {
                chunk[live - 1] &= (1 << (cols % 8)) - 1;
            }
        }
        let fresh = BitMatrix::from_rows(rows, cols, &refill).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                let bit = refill[r * live + c / 8] & (1 << (c % 8)) != 0;
                reclaimed.set(r, c, bit);
            }
        }
        let rank_reclaimed = Ple::decompose(reclaimed, &mut PleScratch::new()).rank();
        let rank_fresh = Ple::decompose(fresh, &mut PleScratch::new()).rank();
        assert_eq!(rank_reclaimed, rank_fresh, "rank at ({rows}, {cols})");
    }
}

#[test]
fn narrow_window_reports_last_column_rank() {
    // Only the last column is nonzero: the window spans two bytes, the
    // panel is narrow, and the masked scan still finds the column.
    let (rows, cols) = (9usize, 15usize);
    let live = cols.div_ceil(8);
    let mut data = vec![0u8; rows * live];
    for r in 0..rows {
        data[r * live + (cols - 1) / 8] |= 1 << ((cols - 1) % 8);
    }
    let matrix = BitMatrix::from_rows(rows, cols, &data).unwrap();
    let ple = Ple::decompose_with_panel_width(matrix, &mut PleScratch::new(), 3);
    assert_eq!(ple.rank(), 1);
    assert_eq!(ple.col_rank_profile(), &[cols - 1]);
    assert_eq!(ple.row_rank_profile(), &[0]);
}

#[test]
fn empty_window_reports_no_pivot() {
    // An all-zero matrix through a narrow panel: every window is
    // empty, the pivot search reports no pivot, and both profiles
    // are empty. A one-column window exercises the same early
    // return through the smallest possible panel width.
    let (rows, cols) = (6usize, 8usize);
    let ple = Ple::decompose_with_panel_width(zeros(rows, cols), &mut PleScratch::new(), 4);
    assert_eq!(ple.rank(), 0);
    assert!(ple.row_rank_profile().is_empty());
    assert!(ple.col_rank_profile().is_empty());
    let ple = Ple::decompose_with_panel_width(zeros(rows, cols), &mut PleScratch::new(), 1);
    assert_eq!(ple.rank(), 0);
}

#[test]
fn exhausted_window_skips_past_dead_columns() {
    // Nonzero content only in column 0 with panel width 1: the first
    // window pivots there, and every later window is dead — the
    // search skips each of them and still names the right profile.
    let (rows, cols) = (5usize, 6usize);
    let live = cols.div_ceil(8);
    let mut bits = vec![0u8; rows * live];
    for r in 0..rows {
        bits[r * live] |= 1;
    }
    let matrix = BitMatrix::from_rows(rows, cols, &bits).unwrap();
    let ple = Ple::decompose_with_panel_width(matrix, &mut PleScratch::new(), 1);
    assert_eq!(ple.rank(), 1);
    assert_eq!(ple.col_rank_profile(), &[0]);
}

#[test]
fn pivot_row_coupling_still_reassembles() {
    // Pivot rows with subdiagonal bits set, narrow panel so the U12
    // triangular solve runs over a real window.
    let (rows, cols) = (10usize, 12usize);
    let mut m = zeros(rows, cols);
    for i in 0..8 {
        m.set(i, i, true);
        for j in 0..i {
            m.set(i, j, (i + j) % 2 == 0);
        }
        m.set(i, 10, true);
        m.set(i, 11, i % 2 == 0);
    }
    let ple = Ple::decompose_with_panel_width(m, &mut PleScratch::new(), 4);
    assert_eq!(ple.rank(), 8);
    // RREF preserves rank and has zeros below every pivot: the
    // pivot columns carry exactly one nonzero each.
    let mut rref = zeros(rows, cols);
    ple.rref_into(&mut rref);
    assert_eq!(
        Ple::decompose(rref.clone(), &mut PleScratch::new()).rank(),
        8
    );
    for &c in ple.col_rank_profile() {
        let weight = (0..rows).filter(|&r| rref.get(r, c)).count();
        assert_eq!(weight, 1, "pivot column {c} weight");
    }
}

/// The byte-for-byte surface, behind `internals`: `lu`, both permutation
/// actions, rank, both profiles, panel-width independence, and the
/// independent `P·L·U·Q` reassembly certificate.
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
