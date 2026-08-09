//! `Ple` against its acceptance: the reassembly certificate, rank agreement
//! with the oracle, rank invariance, kernel/inverse/solve/det/rref checks,
//! and — behind `internals` — byte-for-byte agreement of `lu`, `p`, `q`,
//! and both profiles across panel widths.

// Index arithmetic across several matrices at once; iterators would obscure it.
#![allow(clippy::needless_range_loop)]

mod common;
mod oracles;

use common::{draw, noise};
use fgf::FieldKernels;
use fgf::field::Elem;
use fgf::{FanPaar8, FanPaar16, FanPaar32, FanPaar64, Field, Gf8, Gf16, Gf32, Gf64};
use gfm::{Matrix, Ple, PleScratch, SolveScratch};
use oracles::{
    Naive, naive_det, naive_identity, naive_mul, naive_noise, naive_with_rank, oracle_ple,
    oracle_ple_packed, oracle_rref, pack, packed_with_rank, reassemble, reassemble_packed,
};

/// Builds a crate matrix from packed rows.
fn matrix_of<F: FieldKernels>(packed: &[Vec<u8>], cols: usize) -> Matrix<F> {
    let flat: Vec<u8> = packed.iter().flatten().copied().collect();
    Matrix::from_rows(packed.len(), cols, &flat).unwrap()
}

/// Packs a naive matrix and builds the crate matrix.
fn crate_matrix<F: FieldKernels>(a: &Naive<F>) -> Matrix<F> {
    let cols = a.first().map_or(0, Vec::len);
    matrix_of::<F>(&pack::<F>(a), cols)
}

/// Reads a crate matrix back into naive rows.
fn naive_of<F: FieldKernels>(m: &Matrix<F>) -> Naive<F> {
    (0..m.rows())
        .map(|r| (0..m.cols()).map(|c| m.get(r, c)).collect())
        .collect()
}

/// A zero naive matrix.
fn naive_zero<F: FieldKernels>(rows: usize, cols: usize) -> Naive<F> {
    (0..rows)
        .map(|_| (0..cols).map(|_| F::Elem::ZERO).collect())
        .collect()
}

/// The public-surface check for one (shape, rank) case: the decomposition
/// exists, its rank agrees with the oracle's, and the oracle's `P·L·U·Q`
/// reassembles to the input — the independent certificate.
fn check_case<F: FieldKernels>(rows: usize, cols: usize, rank: usize, seed: u64) {
    let a = packed_with_rank::<F>(rows, cols, rank, seed);
    let ple = Ple::decompose(matrix_of::<F>(&a, cols), &mut PleScratch::new());
    let o = oracle_ple_packed::<F>(&a, cols);
    assert_eq!(ple.rank(), o.rank, "rank at ({rows}, {cols}, r{rank})");
    assert_eq!(
        reassemble_packed::<F>(&o, cols),
        a,
        "P·L·U·Q == A at ({rows}, {cols}, r{rank})"
    );
    #[cfg(feature = "internals")]
    byte_for_byte::assert_matches_oracle_packed(&ple, &o, rows, cols);
}

fn check_case_all_fields(rows: usize, cols: usize, rank: usize, seed: u64) {
    check_case::<Gf8>(rows, cols, rank, seed);
    check_case::<Gf16>(rows, cols, rank, seed);
    check_case::<Gf32>(rows, cols, rank, seed);
    check_case::<Gf64>(rows, cols, rank, seed);
    check_case::<FanPaar8>(rows, cols, rank, seed);
    check_case::<FanPaar16>(rows, cols, rank, seed);
    check_case::<FanPaar32>(rows, cols, rank, seed);
    check_case::<FanPaar64>(rows, cols, rank, seed);
}

/// Rank samples covering both ends and the middle.
fn rank_samples(m: usize, n: usize) -> Vec<usize> {
    let min = m.min(n);
    let mut v: Vec<usize> = [0, 1, 2, min / 2, min.saturating_sub(1), min]
        .into_iter()
        .filter(|&r| r <= min)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn certificate_small_shapes_every_rank() {
    // Every shape 1..=16 x 1..=16, every rank, every field.
    for rows in 1..=16usize {
        for cols in 1..=16usize {
            for rank in 0..=rows.min(cols) {
                check_case_all_fields(rows, cols, rank, 0xA000 + ((rows << 8 | cols) as u64));
            }
        }
    }
}

#[test]
fn scalar_oracle_deep_check() {
    // The fully kernel-independent scalar oracle, every rank at every small
    // shape: the structural ground truth the packed oracle is validated
    // against in `oracles`'s self-check.
    for rows in 1..=12usize {
        for cols in 1..=12usize {
            for rank in 0..=rows.min(cols) {
                let a = naive_with_rank::<Gf8>(
                    rows,
                    cols,
                    rank,
                    0xA800 + ((rows << 8 | cols | rank << 16) as u64),
                );
                let o = oracle_ple::<Gf8>(&a);
                let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
                assert_eq!(ple.rank(), o.rank, "rank at ({rows}, {cols}, r{rank})");
                assert_eq!(
                    reassemble(&o),
                    a,
                    "certificate at ({rows}, {cols}, r{rank})"
                );
            }
        }
    }
}

#[test]
fn certificate_sweep() {
    // Every shape 1..=64 x 1..=64 at sampled ranks.
    for rows in 1..=64usize {
        for cols in 1..=64usize {
            for &rank in &rank_samples(rows, cols) {
                check_case::<Gf8>(
                    rows,
                    cols,
                    rank,
                    0xC000 + ((rows << 8 | cols | rank << 16) as u64),
                );
            }
        }
    }
}

#[test]
fn certificate_sweep_gf16() {
    for rows in 1..=64usize {
        for cols in 1..=64usize {
            for &rank in &rank_samples(rows, cols) {
                check_case::<Gf16>(
                    rows,
                    cols,
                    rank,
                    0xD000 + ((rows << 8 | cols | rank << 16) as u64),
                );
            }
        }
    }
}

#[test]
fn certificate_wide_and_fanpaar_grid() {
    // Coarser lattice for the wide and Fan-Paar fields: the full small
    // square, then a step lattice to 64 with every pitch boundary included.
    let points: Vec<usize> = (1..=16)
        .chain([24, 31, 32, 33, 40, 48, 56, 63, 64])
        .collect();
    for &rows in &points {
        for &cols in &points {
            let samples = rank_samples(rows, cols);
            for &rank in samples.iter().take(3) {
                let seed = (rows << 8 | cols | rank << 16) as u64;
                check_case::<Gf32>(rows, cols, rank, 0xE000 + seed);
                check_case::<Gf64>(rows, cols, rank, 0xE100 + seed);
                check_case::<FanPaar8>(rows, cols, rank, 0xE200 + seed);
                check_case::<FanPaar16>(rows, cols, rank, 0xE300 + seed);
                check_case::<FanPaar32>(rows, cols, rank, 0xE400 + seed);
                check_case::<FanPaar64>(rows, cols, rank, 0xE500 + seed);
            }
        }
    }
}

#[test]
fn certificate_rectangular() {
    for &(rows, cols) in &[
        (256, 64),
        (64, 256),
        (255, 65),
        (65, 255),
        (256, 1),
        (1, 256),
    ] {
        for &rank in &rank_samples(rows, cols) {
            let seed = (rows << 10 | cols | rank << 20) as u64;
            check_case::<Gf8>(rows, cols, rank, 0xF000 + seed);
            check_case::<Gf16>(rows, cols, rank, 0xF100 + seed);
        }
    }
}

#[test]
fn rank_is_invariant() {
    // Rank is invariant under row/column permutation, row scaling, and row
    // addition — checked against the oracle on both sides of each transform.
    let mut state = 0x1A11_u64;
    for case in 0..40 {
        let rows = 1 + draw(&mut state, 24);
        let cols = 1 + draw(&mut state, 24);
        let rank = draw(&mut state, rows.min(cols) + 1);
        let a = naive_with_rank::<Gf16>(rows, cols, rank, 0x1A00 + case);
        assert_eq!(
            oracle_ple::<Gf16>(&a).rank,
            rank,
            "construction is exact-rank"
        );
        let crate_rank = |m: &Naive<Gf16>| {
            Ple::decompose(crate_matrix::<Gf16>(m), &mut PleScratch::new()).rank()
        };

        // Row permutation.
        let mut idx: Vec<usize> = (0..rows).collect();
        for i in 0..rows {
            let j = i + draw(&mut state, rows - i);
            idx.swap(i, j);
        }
        let permuted: Naive<Gf16> = idx.iter().map(|&src| a[src].clone()).collect();
        assert_eq!(crate_rank(&permuted), rank, "row permutation");

        // Column permutation.
        let mut idx: Vec<usize> = (0..cols).collect();
        for i in 0..cols {
            let j = i + draw(&mut state, cols - i);
            idx.swap(i, j);
        }
        let permuted: Naive<Gf16> = a
            .iter()
            .map(|row| idx.iter().map(|&src| row[src]).collect())
            .collect();
        assert_eq!(crate_rank(&permuted), rank, "column permutation");

        // Row scaling by a nonzero scalar.
        let mut scaled = a.clone();
        let target = draw(&mut state, rows);
        let mut s = Gf16::read(&noise(2, 0x5CA1));
        if s.is_zero() {
            s = fgf::gf16::Elem::ONE;
        }
        for cell in scaled[target].iter_mut() {
            *cell = cell.mul(s);
        }
        assert_eq!(crate_rank(&scaled), rank, "row scaling");

        // Row addition.
        if rows > 1 {
            let mut added = a.clone();
            let (dst, src) = (draw(&mut state, rows), draw(&mut state, rows));
            if dst != src {
                for c in 0..cols {
                    added[dst][c] = added[dst][c].add(added[src][c]);
                }
                assert_eq!(crate_rank(&added), rank, "row addition");
            }
        }
    }
}

#[test]
fn det_matches_cofactor() {
    for n in 1..=6usize {
        for rank in 0..=n {
            let a = naive_with_rank::<Gf16>(n, n, rank, 0xDE00 + ((n << 4 | rank) as u64));
            let ple = Ple::decompose(crate_matrix::<Gf16>(&a), &mut PleScratch::new());
            let expected = naive_det::<Gf16>(&a);
            assert_eq!(ple.det(), expected, "det at n={n}, rank {rank}");
            assert_eq!(ple.det().is_zero(), rank < n, "det zero iff rank < n");
        }
    }
}

#[test]
fn kernel_basis_is_a_kernel() {
    let mut state = 0xBE12_u64;
    for case in 0..30 {
        let rows = 1 + draw(&mut state, 24);
        let cols = rows + draw(&mut state, 16);
        let rank = draw(&mut state, rows.min(cols) + 1);
        let a = naive_with_rank::<Gf8>(rows, cols, rank, 0xBE00 + case);
        let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
        let mut kernel = Matrix::<Gf8>::zeros(cols, cols - rank).unwrap();
        ple.kernel_into(&mut kernel);
        // A·K == 0, by naive multiply.
        let product = naive_mul::<Gf8>(&a, &naive_of::<Gf8>(&kernel));
        assert!(
            product.iter().flatten().all(|&v| v.is_zero()),
            "A·kernel == 0 at case {case}"
        );
        assert_eq!(
            ple.rank() + kernel.cols(),
            cols,
            "rank + kernel_dim == cols"
        );
        // The basis itself has full column rank.
        let kple = Ple::decompose(kernel, &mut PleScratch::new());
        assert_eq!(kple.rank(), cols - rank, "kernel basis full rank");
    }
}

#[test]
fn inverse_round_trips() {
    for n in 1..=24usize {
        for rank in [n, n / 2, n.saturating_sub(1)] {
            let a = naive_with_rank::<Gf16>(n, n, rank, 0x1B00 + ((n << 6 | rank) as u64));
            let ple = Ple::decompose(crate_matrix::<Gf16>(&a), &mut PleScratch::new());
            let mut inv = Matrix::<Gf16>::zeros(n, n).unwrap();
            let result = ple.inverse_into(&mut inv);
            if rank < n {
                assert_eq!(
                    result,
                    Err(gfm::SolveError::Singular { rank, order: n }),
                    "Singular names rank and order at n={n}"
                );
                continue;
            }
            result.unwrap();
            let product = naive_mul::<Gf16>(&a, &naive_of::<Gf16>(&inv));
            assert_eq!(product, naive_identity::<Gf16>(n), "A·A⁻¹ == I at n={n}");
        }
    }
}

#[test]
fn inverse_leaves_output_untouched_on_error() {
    let a = naive_with_rank::<Gf8>(6, 6, 3, 0x5151);
    let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
    let mut out = crate_matrix::<Gf8>(&naive_noise::<Gf8>(6, 6, 0x5EA7));
    let before = out.clone();
    assert!(ple.inverse_into(&mut out).is_err());
    assert_eq!(out, before, "output untouched on Singular");
}

#[test]
fn solve_consistent_systems() {
    let mut state = 0x5015E_u64;
    for case in 0..30 {
        let n = 1 + draw(&mut state, 20);
        let rank = draw(&mut state, n + 1);
        let a = naive_with_rank::<Gf8>(n, n, rank, 0x50A0 + case);
        let x0 = naive_noise::<Gf8>(n, 2, 0x50B0 + case);
        let b = naive_mul::<Gf8>(&a, &x0);
        let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
        let mut x = Matrix::<Gf8>::zeros(n, 2).unwrap();
        ple.solve_into(&crate_matrix::<Gf8>(&b), &mut x, &mut SolveScratch::new())
            .unwrap();
        // Any solution is acceptable: check A·x == b.
        let product = naive_mul::<Gf8>(&a, &naive_of::<Gf8>(&x));
        assert_eq!(product, b, "A·x == b at case {case} (n={n}, rank {rank})");
    }
}
#[test]
fn solve_undoes_multiple_column_swaps_in_reverse() {
    let zero = <Gf8 as Field>::Elem::ZERO;
    let one = <Gf8 as Field>::Elem::ONE;
    let a = vec![
        vec![zero, one, zero, one],
        vec![zero, zero, one, one],
        vec![zero, one, one, zero],
        vec![zero, zero, zero, zero],
        vec![zero, one, zero, one],
    ];
    let x0 = naive_noise::<Gf8>(4, 2, 0xC01A);
    let b = naive_mul::<Gf8>(&a, &x0);
    let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
    assert_eq!(ple.rank(), 2);
    let mut x = Matrix::<Gf8>::zeros(4, 2).unwrap();
    ple.solve_into(&crate_matrix::<Gf8>(&b), &mut x, &mut SolveScratch::new())
        .unwrap();
    assert_eq!(naive_mul::<Gf8>(&a, &naive_of::<Gf8>(&x)), b);

    let mut kernel = Matrix::<Gf8>::zeros(4, 2).unwrap();
    ple.kernel_into(&mut kernel);
    assert!(
        naive_mul::<Gf8>(&a, &naive_of::<Gf8>(&kernel))
            .iter()
            .flatten()
            .all(|value| value.is_zero())
    );
}

#[test]
fn solve_reports_genuine_inconsistency() {
    // A with a dependent row, and b that breaks the dependence: the system
    // is genuinely inconsistent, and the named row must be in the
    // eliminated tail. State is preserved on the error path.
    let mut a = naive_with_rank::<Gf8>(8, 6, 5, 0x1C00);
    for c in 0..6 {
        a[7][c] = a[0][c].add(a[1][c]);
    }
    let mut b = naive_noise::<Gf8>(8, 1, 0x1C01);
    b[7][0] = b[0][0].add(b[1][0]).add(fgf::gf8::Elem::ONE);
    let ple = Ple::decompose(crate_matrix::<Gf8>(&a), &mut PleScratch::new());
    let mut x = crate_matrix::<Gf8>(&naive_noise::<Gf8>(6, 1, 0x1C02));
    let before = x.clone();
    let err = ple
        .solve_into(&crate_matrix::<Gf8>(&b), &mut x, &mut SolveScratch::new())
        .unwrap_err();
    let gfm::SolveError::Inconsistent { row } = err else {
        panic!("expected Inconsistent, got {err:?}");
    };
    // Genuinely inconsistent: the augmented matrix has higher rank.
    let augmented: Naive<Gf8> = a
        .iter()
        .zip(&b)
        .map(|(row, bcell)| {
            let mut row = row.clone();
            row.push(bcell[0]);
            row
        })
        .collect();
    let rank_a = oracle_ple::<Gf8>(&a).rank;
    let rank_ab = oracle_ple::<Gf8>(&augmented).rank;
    assert!(rank_ab > rank_a, "the system is genuinely inconsistent");
    assert!(row >= rank_a, "named row is in the eliminated tail");
    assert_eq!(x, before, "state unchanged on Inconsistent");
}

#[test]
fn rref_matches_oracle() {
    for rows in 1..=10usize {
        for cols in 1..=10usize {
            for rank in [0, rows.min(cols) / 2, rows.min(cols)] {
                let seed = (rows << 8 | cols | rank << 16) as u64;
                let a = naive_with_rank::<Gf32>(rows, cols, rank, 0x2200 + seed);
                let ple = Ple::decompose(crate_matrix::<Gf32>(&a), &mut PleScratch::new());
                let mut out = Matrix::<Gf32>::zeros(rows, cols).unwrap();
                ple.rref_into(&mut out);
                assert_eq!(
                    naive_of::<Gf32>(&out),
                    oracle_rref::<Gf32>(&a),
                    "rref at ({rows}, {cols}, r{rank})"
                );
            }
        }
    }
}

#[test]
fn gemm_matches_naive() {
    let a = naive_noise::<Gf16>(9, 7, 0x6E11);
    let b = naive_noise::<Gf16>(7, 5, 0x6E12);
    let mut out = Matrix::<Gf16>::zeros(9, 5).unwrap();
    gfm::mul_into(
        &mut out,
        &crate_matrix::<Gf16>(&a),
        &crate_matrix::<Gf16>(&b),
    );
    assert_eq!(naive_of::<Gf16>(&out), naive_mul::<Gf16>(&a, &b));
    // Accumulating form: A·B + A·B cancels in characteristic two.
    gfm::mul_add_into(
        &mut out,
        &crate_matrix::<Gf16>(&a),
        &crate_matrix::<Gf16>(&b),
    );
    assert_eq!(naive_of::<Gf16>(&out), naive_zero::<Gf16>(9, 5));
}

#[test]
fn trsm_matches_naive() {
    // Unit lower: L·X = B.
    let mut l = naive_noise::<Gf8>(8, 8, 0x7A11);
    for i in 0..8 {
        for t in 0..8 {
            l[i][t] = if t < i {
                l[i][t]
            } else if t == i {
                fgf::gf8::Elem::ONE
            } else {
                fgf::gf8::Elem::ZERO
            };
        }
    }
    let b = naive_noise::<Gf8>(8, 3, 0x7A12);
    let mut x = crate_matrix::<Gf8>(&b);
    gfm::solve_lower_unit_into(&crate_matrix::<Gf8>(&l), &mut x);
    assert_eq!(naive_mul::<Gf8>(&l, &naive_of::<Gf8>(&x)), b, "lower solve");
    // Upper with nonzero diagonal.
    let mut u = naive_noise::<Gf8>(8, 8, 0x7A13);
    for i in 0..8 {
        for t in 0..8 {
            if t < i {
                u[i][t] = fgf::gf8::Elem::ZERO;
            } else if t == i && u[i][t].is_zero() {
                u[i][t] = fgf::gf8::Elem::ONE;
            }
        }
    }
    let mut x = crate_matrix::<Gf8>(&b);
    gfm::solve_upper_into(&crate_matrix::<Gf8>(&u), &mut x);
    assert_eq!(naive_mul::<Gf8>(&u, &naive_of::<Gf8>(&x)), b, "upper solve");
}

/// The byte-for-byte surface, behind `internals`: `lu`, both permutations,
/// both profiles, and panel-width independence.
#[cfg(feature = "internals")]
mod byte_for_byte {
    use super::*;
    use oracles::{PackedPle, apply_swap_list, profile_of};

    /// Asserts the crate's decomposition matches the packed oracle
    /// byte-for-byte: `lu` contents, both permutation actions, rank, and
    /// both profiles.
    pub(super) fn assert_matches_oracle_packed<F: FieldKernels>(
        ple: &Ple<F>,
        o: &PackedPle,
        m: usize,
        n: usize,
    ) {
        assert_eq!(ple.rank(), o.rank, "rank");
        let lu = ple.lu();
        for i in 0..m {
            assert_eq!(lu.row(i), &o.lu[i][..], "lu row {i}");
        }
        let mut probe: Vec<usize> = (0..m).collect();
        ple.p().apply(&mut probe);
        assert_eq!(probe, apply_swap_list(&o.p, m), "row permutation");
        let mut probe: Vec<usize> = (0..n).collect();
        ple.q().apply(&mut probe);
        assert_eq!(probe, apply_swap_list(&o.q, n), "column permutation");
        assert_eq!(
            ple.row_rank_profile(),
            &profile_of(&o.p, m, o.rank)[..],
            "row rank profile"
        );
        assert_eq!(
            ple.col_rank_profile(),
            &profile_of(&o.q, n, o.rank)[..],
            "column rank profile"
        );
    }

    #[test]
    fn panel_widths_agree_byte_for_byte() {
        // The blocked result is independent of the panel width, byte for
        // byte, on every shape — including rank-deficient panels.
        for (rows, cols, rank) in [
            (17, 15, 12),
            (16, 16, 16),
            (33, 31, 9),
            (24, 40, 0),
            (40, 24, 24),
        ] {
            let a = packed_with_rank::<Gf8>(
                rows,
                cols,
                rank,
                0x9A00 + ((rows << 8 | cols | rank << 16) as u64),
            );
            let o = oracle_ple_packed::<Gf8>(&a, cols);
            for width in [1, 2, 3, 5, 7, 64] {
                let ple = Ple::decompose_with_panel_width(
                    matrix_of::<Gf8>(&a, cols),
                    &mut PleScratch::new(),
                    width,
                );
                assert_matches_oracle_packed(&ple, &o, rows, cols);
            }
        }
    }
    #[test]
    fn newton_john_agrees_byte_for_byte() {
        for (rows, cols, rank) in [
            (32, 32, 32),
            (128, 128, 91),
            (200, 130, 113),
            (130, 200, 79),
        ] {
            let a = packed_with_rank::<Gf8>(
                rows,
                cols,
                rank,
                0x6A00 ^ ((rows << 8 | cols | rank << 20) as u64),
            );
            let oracle = oracle_ple_packed::<Gf8>(&a, cols);
            let ple =
                Ple::decompose_newton_john(matrix_of::<Gf8>(&a, cols), &mut PleScratch::new());
            assert_matches_oracle_packed(&ple, &oracle, rows, cols);
        }
    }
}

/// The one-elimination invariant, checked mechanically: the pivot search
/// `find_pivot` appears in `dense/ple.rs` and nowhere else in the crate.
#[test]
fn one_pivot_loop_only() {
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
                if text.contains("find_pivot") {
                    offenders.push(path);
                }
            }
        }
    }
    assert_eq!(
        offenders,
        [std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("dense")
            .join("ple.rs")],
        "a second pivot loop exists"
    );
}
