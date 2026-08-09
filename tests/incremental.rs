//! The streaming accumulator against its acceptance: agreement with a batch
//! `Ple` over the same rows (rank, pivot columns, solution), order-
//! independence, and inconsistency detection.

#![allow(clippy::needless_range_loop)]

mod common;

use common::{draw, noise};
use fgf::Gf8;
use fgf::field::Field;
use gfm::{Cauchy, Echelon, Innovation, Matrix, Ple, PleScratch, SolveScratch};

type E = <Gf8 as Field>::Elem;

/// `n` field elements from deterministic noise.
fn elems(n: usize, seed: u64) -> Vec<E> {
    let bytes = noise(n, seed);
    bytes.iter().map(|&b| Gf8::read(&[b])).collect()
}

/// Packs a row of elements into little-endian bytes.
fn pack(row: &[E]) -> Vec<u8> {
    let mut out = vec![0u8; row.len()];
    for (i, &e) in row.iter().enumerate() {
        Gf8::write(&mut out[i..=i], e);
    }
    out
}

/// A dense `Matrix<Gf8>` from rows of elements.
fn matrix_of(rows: &[Vec<E>], cols: usize) -> Matrix<Gf8> {
    let mut m = Matrix::<Gf8>::zeros(rows.len(), cols).unwrap();
    for (i, row) in rows.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            m.set(i, j, e);
        }
    }
    m
}

#[test]
fn recovered_matches_the_unique_solution() {
    for n in [1usize, 2, 5, 16, 33] {
        let s = 3; // symbols per variable
        // A guaranteed-invertible coefficient matrix.
        let cauchy = Cauchy::<Gf8>::indexed(n, n).unwrap();
        let coeffs: Vec<Vec<E>> = (0..n)
            .map(|i| (0..n).map(|j| cauchy.coeff(i, j)).collect())
            .collect();
        // A random true solution X (n × s) and the induced RHS = C·X.
        let x: Vec<Vec<E>> = (0..n).map(|v| elems(s, 0xA000 + (v as u64))).collect();
        let rhs: Vec<Vec<E>> = (0..n)
            .map(|i| {
                (0..s)
                    .map(|c| {
                        let mut acc = E::ZERO;
                        for j in 0..n {
                            acc = acc.add(coeffs[i][j].mul(x[j][c]));
                        }
                        acc
                    })
                    .collect()
            })
            .collect();

        let mut ech = Echelon::<Gf8>::new(n, s, true).unwrap();
        for i in 0..n {
            ech.absorb(&pack(&coeffs[i]), &pack(&rhs[i]));
        }
        assert!(ech.is_complete(), "n={n}");
        assert_eq!(ech.rank(), n);

        let recovered: std::collections::HashMap<usize, Vec<u8>> = ech
            .recovered()
            .map(|(c, bytes)| (c, bytes.to_vec()))
            .collect();
        assert_eq!(recovered.len(), n, "every variable recovered, n={n}");
        for v in 0..n {
            assert_eq!(recovered[&v], pack(&x[v]), "variable {v} value, n={n}");
        }
    }
}

#[test]
fn rank_and_pivots_agree_with_ple() {
    for &(num_rows, cols) in &[(10, 8), (8, 12), (20, 16), (16, 16), (5, 5)] {
        let rows: Vec<Vec<E>> = (0..num_rows)
            .map(|i| elems(cols, 0xB000 + (i as u64)))
            .collect();
        // Homogeneous RHS keeps the system consistent, isolating rank/pivots.
        let mut ech = Echelon::<Gf8>::new(cols, 1, true).unwrap();
        let zero = vec![0u8; 1];
        for row in &rows {
            ech.absorb(&pack(row), &zero);
        }
        let ple = Ple::decompose(matrix_of(&rows, cols), &mut PleScratch::new());
        assert_eq!(ech.rank(), ple.rank(), "rank ({num_rows}x{cols})");
        let pivots: Vec<_> = ech.pivot_columns().collect();
        assert_eq!(
            pivots,
            ple.col_rank_profile(),
            "pivot columns ({num_rows}x{cols})"
        );
    }
}

#[test]
fn absorb_order_is_irrelevant() {
    // A rank-deficient but consistent system: recovered set, rank, and
    // inconsistency verdict are invariant under absorb order.
    let (num_rows, cols, s) = (14usize, 10usize, 2usize);
    // Build rows with deliberate dependence: first 6 independent, rest combos.
    let base: Vec<Vec<E>> = (0..6).map(|i| elems(cols, 0xC000 + i as u64)).collect();
    let x: Vec<Vec<E>> = (0..cols).map(|v| elems(s, 0xD000 + v as u64)).collect();
    let mut rows: Vec<Vec<E>> = base.clone();
    let mut st = 0xE1E1u64;
    while rows.len() < num_rows {
        // A random XOR of two base rows — dependent, so rank stays <= 6.
        let a = draw(&mut st, base.len());
        let b = draw(&mut st, base.len());
        rows.push((0..cols).map(|c| base[a][c].add(base[b][c])).collect());
    }
    let rhs: Vec<Vec<E>> = rows
        .iter()
        .map(|row| {
            (0..s)
                .map(|c| {
                    let mut acc = E::ZERO;
                    for j in 0..cols {
                        acc = acc.add(row[j].mul(x[j][c]));
                    }
                    acc
                })
                .collect()
        })
        .collect();

    let run = |order: &[usize]| -> (usize, Vec<(usize, Vec<u8>)>, bool) {
        let mut ech = Echelon::<Gf8>::new(cols, s, true).unwrap();
        let mut inconsistent = false;
        for &i in order {
            if matches!(
                ech.absorb(&pack(&rows[i]), &pack(&rhs[i])),
                Innovation::Inconsistent
            ) {
                inconsistent = true;
            }
        }
        let mut rec: Vec<(usize, Vec<u8>)> =
            ech.recovered().map(|(c, b)| (c, b.to_vec())).collect();
        rec.sort_by_key(|(c, _)| *c);
        (ech.rank(), rec, inconsistent)
    };

    let natural: Vec<usize> = (0..num_rows).collect();
    let mut shuffled = natural.clone();
    // A deterministic shuffle.
    let mut st = 0xF00Du64;
    for i in (1..num_rows).rev() {
        shuffled.swap(i, draw(&mut st, i + 1));
    }
    let reversed: Vec<usize> = (0..num_rows).rev().collect();

    let a = run(&natural);
    let b = run(&shuffled);
    let c = run(&reversed);
    assert_eq!(a, b, "natural vs shuffled");
    assert_eq!(a, c, "natural vs reversed");
    assert!(!a.2, "consistent system flagged inconsistent");
}

#[test]
fn detects_inconsistency() {
    let cols = 6;
    let mut ech = Echelon::<Gf8>::new(cols, 1, true).unwrap();
    let row = elems(cols, 0x1234);
    // First absorb is innovative.
    assert!(matches!(
        ech.absorb(&pack(&row), &pack(&[Gf8::read(&[7])])),
        Innovation::Innovative { .. }
    ));
    // The same coefficients with a different RHS contradict it.
    let verdict = ech.absorb(&pack(&row), &pack(&[Gf8::read(&[9])]));
    assert_eq!(verdict, Innovation::Inconsistent);
    // The same coefficients with the same RHS are merely dependent.
    let verdict = ech.absorb(&pack(&row), &pack(&[Gf8::read(&[7])]));
    assert_eq!(verdict, Innovation::Dependent);
}

#[test]
fn forward_echelon_matches_reduced_rank() {
    // The recoder (reduced == false) and decoder agree on rank, and on the
    // final solution when driven to completion via a batch Ple.
    for &(num_rows, cols) in &[(12, 10), (8, 8), (20, 15)] {
        let rows: Vec<Vec<E>> = (0..num_rows)
            .map(|i| elems(cols, 0x2200 + i as u64))
            .collect();
        let zero = vec![0u8; 1];
        let mut decoder = Echelon::<Gf8>::new(cols, 1, true).unwrap();
        let mut recoder = Echelon::<Gf8>::new(cols, 1, false).unwrap();
        for row in &rows {
            decoder.absorb(&pack(row), &zero);
            recoder.absorb(&pack(row), &zero);
        }
        assert_eq!(decoder.rank(), recoder.rank(), "{num_rows}x{cols}");
        let ple = Ple::decompose(matrix_of(&rows, cols), &mut PleScratch::new());
        assert_eq!(decoder.rank(), ple.rank());
    }
}

#[test]
fn solution_matches_ple_solve_when_complete() {
    // Cross-check recovered payloads against Ple::solve_into on the same rows.
    let (n, s) = (12usize, 4usize);
    let cauchy = Cauchy::<Gf8>::indexed(n, n).unwrap();
    let coeffs: Vec<Vec<E>> = (0..n)
        .map(|i| (0..n).map(|j| cauchy.coeff(i, j)).collect())
        .collect();
    let x: Vec<Vec<E>> = (0..n).map(|v| elems(s, 0x3300 + v as u64)).collect();
    let rhs: Vec<Vec<E>> = (0..n)
        .map(|i| {
            (0..s)
                .map(|c| {
                    let mut acc = E::ZERO;
                    for j in 0..n {
                        acc = acc.add(coeffs[i][j].mul(x[j][c]));
                    }
                    acc
                })
                .collect()
        })
        .collect();

    let mut ech = Echelon::<Gf8>::new(n, s, true).unwrap();
    for i in 0..n {
        ech.absorb(&pack(&coeffs[i]), &pack(&rhs[i]));
    }

    let a = matrix_of(&coeffs, n);
    let b = matrix_of(&rhs, s);
    let ple = Ple::decompose(a, &mut PleScratch::new());
    let mut out = Matrix::<Gf8>::zeros(n, s).unwrap();
    ple.solve_into(&b, &mut out, &mut SolveScratch::new())
        .unwrap();

    for (c, bytes) in ech.recovered() {
        assert_eq!(bytes, out.row(c), "variable {c}");
    }
}

#[test]
fn advancing_prefix_reindexes_surviving_rows() {
    for reduced in [false, true] {
        let mut echelon = Echelon::<Gf8>::new(5, 1, reduced).unwrap();
        assert!(matches!(
            echelon.absorb(&[1, 0, 0, 0, 0], &[11]),
            Innovation::Innovative { pivot: 0 }
        ));
        assert!(matches!(
            echelon.absorb(&[0, 0, 1, 0, 7], &[22]),
            Innovation::Innovative { pivot: 2 }
        ));
        assert!(matches!(
            echelon.absorb(&[0, 0, 0, 0, 1], &[33]),
            Innovation::Innovative { pivot: 4 }
        ));

        echelon.advance_prefix(2);
        assert_eq!(echelon.rank(), 2);
        assert_eq!(echelon.pivot_columns().collect::<Vec<_>>(), [0, 2]);
        let rows: Vec<_> = echelon
            .retained_rows()
            .map(|row| {
                let pivot = row.pivot();
                let mut coefficients = vec![0; 5];
                row.mul_add_coefficients_into(&mut coefficients, Gf8::read(&[1]));
                let (first, second) = row.coefficient_slices();
                let packed: Vec<_> = first.iter().chain(second).copied().collect();
                let last = coefficients
                    .iter()
                    .rposition(|&coefficient| coefficient != 0)
                    .expect("retained row has a pivot");
                assert_eq!(packed, coefficients[pivot..=last]);
                (pivot, coefficients, row.rhs().to_vec())
            })
            .collect();
        assert_eq!(rows.iter().map(|row| row.0).collect::<Vec<_>>(), [0, 2]);
        assert!(
            rows.iter()
                .all(|row| row.1[..row.0].iter().all(|&x| x == 0))
        );

        assert!(matches!(
            echelon.absorb(&[0, 0, 0, 0, 1], &[44]),
            Innovation::Innovative { pivot: 4 }
        ));
        assert_eq!(echelon.rank(), 3);
        echelon.advance_prefix(5);
        assert_eq!(echelon.rank(), 0);
        assert!(echelon.retained_rows().next().is_none());
    }
}
