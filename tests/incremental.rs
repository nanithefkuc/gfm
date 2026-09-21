//! The streaming accumulator against its acceptance: agreement with a batch
//! `Ple` over the same rows in rank, pivot columns, and solution, order
//! independence, inconsistency detection, pivot queries, ragged and wide
//! geometry, ring wrap after a prefix advance, and reduced-mode substitution
//! of recovered columns.

// Index arithmetic across several matrices at once; iterators would obscure it.
#![allow(clippy::needless_range_loop)]

mod common;

use common::{draw, noise};
use fgf::field::{Elem, Field};
use fgf::{FieldKernels, Gf8B, Gf16};
use gfm::{Echelon, Innovation, Matrix, Ple, PleScratch, SolveScratch};

type E = <Gf8B as Field>::Elem;

/// Draws `n` arbitrary field elements deterministically.
fn elems<F: FieldKernels>(n: usize, seed: u64) -> Vec<F::Elem> {
    let bytes = noise(n * F::BYTES, seed);
    (0..n)
        .map(|i| F::decode(&bytes[i * F::BYTES..(i + 1) * F::BYTES]))
        .collect()
}

/// Packs a row of elements into little-endian bytes.
fn pack<F: FieldKernels>(row: &[F::Elem]) -> Vec<u8> {
    let mut out = vec![0u8; row.len() * F::BYTES];
    for (i, &e) in row.iter().enumerate() {
        F::encode(&mut out[i * F::BYTES..(i + 1) * F::BYTES], e);
    }
    out
}

/// A dense `Matrix<Gf8B>` from rows of elements.
fn matrix_of(rows: &[Vec<E>], cols: usize) -> Matrix<Gf8B> {
    let mut m = Matrix::<Gf8B>::zeros(rows.len(), cols).unwrap();
    for (i, row) in rows.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            m.set(i, j, e);
        }
    }
    m
}

/// A guaranteed-invertible dense coefficient matrix: the product `L·U` of a
/// unit-diagonal lower triangle and a unit-diagonal upper triangle filled
/// from deterministic noise, so it is nonsingular and generically dense.
fn invertible_coeffs<F: FieldKernels>(n: usize, seed: u64) -> Vec<Vec<F::Elem>> {
    let entry = |i: usize, j: usize| elems::<F>(1, seed + ((i * n + j) as u64) * 0x9E37)[0];
    let mut a = vec![vec![F::Elem::ZERO; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = F::Elem::ZERO;
            for t in 0..n {
                let l = if t < i {
                    entry(i, t)
                } else if t == i {
                    F::Elem::ONE
                } else {
                    F::Elem::ZERO
                };
                let u = if t > j {
                    entry(t, j)
                } else if t == j {
                    F::Elem::ONE
                } else {
                    F::Elem::ZERO
                };
                acc = acc.add(l.mul(u));
            }
            *cell = acc;
        }
    }
    a
}

/// The same invertible matrix with a nonzero trailing entry in every row, so
/// every absorption also exercises the substitution loop past the pivot: unit
/// rows recover first, later rows substitute them out through the kernel path.
fn invertible_with_tail(n: usize, seed: u64) -> Vec<Vec<E>> {
    let mut a = invertible_coeffs::<Gf8B>(n, seed);
    for row in &mut a {
        if row[n - 1].is_zero() {
            row[n - 1] = E::ONE;
        }
    }
    a
}

#[test]
fn recovered_matches_the_unique_solution() {
    for n in [1usize, 2, 5, 16, 33] {
        let s = 3; // symbols per variable
        // A guaranteed-invertible coefficient matrix.
        let coeffs = invertible_coeffs::<Gf8B>(n, 0xA000);
        // A random true solution X (n × s) and the induced RHS = C·X.
        let x: Vec<Vec<E>> = (0..n)
            .map(|v| elems::<Gf8B>(s, 0xA000 + (v as u64)))
            .collect();
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

        let mut ech = Echelon::<Gf8B>::new(n, s, true).unwrap();
        for i in 0..n {
            ech.absorb(&pack::<Gf8B>(&coeffs[i]), &pack::<Gf8B>(&rhs[i]));
        }
        assert!(ech.is_complete(), "n={n}");
        assert_eq!(ech.rank(), n);

        let recovered: std::collections::HashMap<usize, Vec<u8>> = ech
            .recovered()
            .map(|(c, bytes)| (c, bytes.to_vec()))
            .collect();
        assert_eq!(recovered.len(), n, "every variable recovered, n={n}");
        for v in 0..n {
            assert_eq!(
                recovered[&v],
                pack::<Gf8B>(&x[v]),
                "variable {v} value, n={n}"
            );
        }
    }
}

#[test]
fn rank_and_pivots_agree_with_ple() {
    for &(num_rows, cols) in &[(10, 8), (8, 12), (20, 16), (16, 16), (5, 5)] {
        let rows: Vec<Vec<E>> = (0..num_rows)
            .map(|i| elems::<Gf8B>(cols, 0xB000 + (i as u64)))
            .collect();
        // Homogeneous RHS keeps the system consistent, isolating rank/pivots.
        let mut ech = Echelon::<Gf8B>::new(cols, 1, true).unwrap();
        let zero = vec![0u8; 1];
        for row in &rows {
            ech.absorb(&pack::<Gf8B>(row), &zero);
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
    let base: Vec<Vec<E>> = (0..6)
        .map(|i| elems::<Gf8B>(cols, 0xC000 + i as u64))
        .collect();
    let x: Vec<Vec<E>> = (0..cols)
        .map(|v| elems::<Gf8B>(s, 0xD000 + v as u64))
        .collect();
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
        let mut ech = Echelon::<Gf8B>::new(cols, s, true).unwrap();
        let mut inconsistent = false;
        for &i in order {
            if matches!(
                ech.absorb(&pack::<Gf8B>(&rows[i]), &pack::<Gf8B>(&rhs[i])),
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
    let mut ech = Echelon::<Gf8B>::new(cols, 1, true).unwrap();
    let row = elems::<Gf8B>(cols, 0x1234);
    // First absorb is innovative.
    assert!(matches!(
        ech.absorb(&pack::<Gf8B>(&row), &pack::<Gf8B>(&[Gf8B::decode(&[7])])),
        Innovation::Innovative { .. }
    ));
    // The same coefficients with a different RHS contradict it.
    let verdict = ech.absorb(&pack::<Gf8B>(&row), &pack::<Gf8B>(&[Gf8B::decode(&[9])]));
    assert_eq!(verdict, Innovation::Inconsistent);
    // The same coefficients with the same RHS are merely dependent.
    let verdict = ech.absorb(&pack::<Gf8B>(&row), &pack::<Gf8B>(&[Gf8B::decode(&[7])]));
    assert_eq!(verdict, Innovation::Dependent);
}

#[test]
fn forward_echelon_matches_reduced_rank() {
    // The recoder (reduced == false) and decoder agree on rank, and on the
    // final solution when driven to completion via a batch Ple.
    for &(num_rows, cols) in &[(12, 10), (8, 8), (20, 15)] {
        let rows: Vec<Vec<E>> = (0..num_rows)
            .map(|i| elems::<Gf8B>(cols, 0x2200 + i as u64))
            .collect();
        let zero = vec![0u8; 1];
        let mut decoder = Echelon::<Gf8B>::new(cols, 1, true).unwrap();
        let mut recoder = Echelon::<Gf8B>::new(cols, 1, false).unwrap();
        for row in &rows {
            decoder.absorb(&pack::<Gf8B>(row), &zero);
            recoder.absorb(&pack::<Gf8B>(row), &zero);
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
    let coeffs = invertible_coeffs::<Gf8B>(n, 0x3300);
    let x: Vec<Vec<E>> = (0..n)
        .map(|v| elems::<Gf8B>(s, 0x3300 + v as u64))
        .collect();
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

    let mut ech = Echelon::<Gf8B>::new(n, s, true).unwrap();
    for i in 0..n {
        ech.absorb(&pack::<Gf8B>(&coeffs[i]), &pack::<Gf8B>(&rhs[i]));
    }

    let a = matrix_of(&coeffs, n);
    let b = matrix_of(&rhs, s);
    let ple = Ple::decompose(a, &mut PleScratch::new());
    let mut out = Matrix::<Gf8B>::zeros(n, s).unwrap();
    ple.solve_into(&b, &mut out, &mut SolveScratch::new())
        .unwrap();

    for (c, bytes) in ech.recovered() {
        assert_eq!(bytes, out.row(c), "variable {c}");
    }
}

#[test]
fn is_pivot_tracks_absorption() {
    let mut echelon = Echelon::<Gf8B>::new(4, 1, false).unwrap();
    for c in 0..4 {
        assert!(!echelon.is_pivot(c), "pivot {c} before absorption");
    }
    assert!(matches!(
        echelon.absorb(&[1, 0, 0, 0], &[3]),
        Innovation::Innovative { pivot: 0 }
    ));
    assert!(echelon.is_pivot(0));
    for c in 1..4 {
        assert!(!echelon.is_pivot(c), "pivot {c} after one absorption");
    }
    assert_eq!(echelon.rank(), 1);
    assert!(!echelon.is_complete());
}

#[test]
fn constructor_rejects_ragged_rhs() {
    let err = Echelon::<Gf16>::new(4, 3, true).unwrap_err();
    assert_eq!(
        err,
        gfm::GeometryError::Ragged {
            len: 3,
            element_bytes: 2,
        }
    );
}

#[test]
fn absorb_unit_matches_a_dense_unit_row() {
    for reduced in [false, true] {
        let mut echelon = Echelon::<Gf8B>::new(4, 1, reduced).unwrap();
        assert!(matches!(
            echelon.absorb_unit(2, &[9]),
            Innovation::Innovative { pivot: 2 }
        ));
        assert_eq!(echelon.rank(), 1);
        assert!(echelon.is_pivot(2));
        // The same absorption through the dense path agrees.
        let mut dense = Echelon::<Gf8B>::new(4, 1, reduced).unwrap();
        assert!(matches!(
            dense.absorb(&[0, 0, 1, 0], &[9]),
            Innovation::Innovative { pivot: 2 }
        ));
        assert_eq!(echelon.pivot_columns().collect::<Vec<_>>(), vec![2]);
        assert_eq!(dense.pivot_columns().collect::<Vec<_>>(), vec![2]);
        if reduced {
            assert_eq!(echelon.recovered_value(2), Some(&[9][..]));
            assert_eq!(echelon.recovered_value(0), None);
        }
        // Both complete to the same solution as a batch Ple.
        let x = elems::<Gf8B>(4, 0x0417);
        let mut unit_then_dense = Echelon::<Gf8B>::new(4, 1, reduced).unwrap();
        unit_then_dense.absorb_unit(2, &pack::<Gf8B>(&[x[2]]));
        let mut absolute: Vec<Vec<E>> = vec![vec![E::ZERO; 4]];
        absolute[0][2] = E::ONE;
        let mut absolute_rhs = vec![vec![x[2]]];
        // Dense completion rows pivoting around the unit: `e_i + e_2` for
        // `i != 2` is triangular after a row permutation, hence full rank,
        // and each row reduces against the retained unit on absorption.
        for i in [0, 1, 3] {
            let mut coefficients = vec![E::ZERO; 4];
            coefficients[i] = E::ONE;
            coefficients[2] = E::ONE;
            let rhs = x[i].add(x[2]);
            unit_then_dense.absorb(&pack::<Gf8B>(&coefficients), &pack::<Gf8B>(&[rhs]));
            absolute.push(coefficients);
            absolute_rhs.push(vec![rhs]);
        }
        assert!(unit_then_dense.is_complete());
        let (a, b) = (
            matrix_of(&absolute, absolute[0].len()),
            matrix_of(&absolute_rhs, 1),
        );
        let ple = Ple::decompose(a, &mut PleScratch::new());
        let mut out = Matrix::<Gf8B>::zeros(4, 1).unwrap();
        ple.solve_into(&b, &mut out, &mut SolveScratch::new())
            .unwrap();
        for v in 0..4 {
            assert_eq!(out.get(v, 0), x[v], "variable {v}");
            if reduced {
                assert_eq!(
                    unit_then_dense.recovered_value(v).unwrap(),
                    out.row(v),
                    "recovered variable {v}"
                );
            }
        }
    }
}

#[test]
fn reduced_absorb_substitutes_recovered_columns() {
    // Units on 2 and 3 recover first; a later row pivoting on 0 with a
    // nonzero entry at 2 substitutes the known value out.
    let x = elems::<Gf8B>(4, 0x5EED);
    let mut echelon = Echelon::<Gf8B>::new(4, 1, true).unwrap();
    echelon.absorb_unit(2, &pack::<Gf8B>(&[x[2]]));
    echelon.absorb_unit(3, &pack::<Gf8B>(&[x[3]]));
    assert_eq!(echelon.newly_recovered_columns(), &[3]);
    let r0 = x[0].add(x[2]);
    assert!(matches!(
        echelon.absorb(&[1, 0, 1, 0], &pack::<Gf8B>(&[r0])),
        Innovation::Innovative { pivot: 0 }
    ));
    // Only column 0 became recovered in the last absorption.
    assert_eq!(echelon.newly_recovered_columns(), &[0]);
    let fresh: Vec<(usize, Vec<u8>)> = echelon
        .newly_recovered()
        .map(|(c, bytes)| (c, bytes.to_vec()))
        .collect();
    assert_eq!(fresh, vec![(0, pack::<Gf8B>(&[x[0]]))]);
    assert_eq!(
        echelon.recovered_value(0).unwrap(),
        &pack::<Gf8B>(&[x[0]])[..]
    );
}

#[test]
fn unit_propagation_shrinks_spans_and_recovers() {
    let x = elems::<Gf8B>(3, 0x9011);
    let mut echelon = Echelon::<Gf8B>::new(3, 1, true).unwrap();
    let r0 = x[0].add(x[1]).add(x[2]);
    assert!(matches!(
        echelon.absorb(&[1, 1, 1], &pack::<Gf8B>(&[r0])),
        Innovation::Innovative { pivot: 0 }
    ));
    assert!(echelon.newly_recovered_columns().is_empty());
    // Recovering column 2 clears the trailing entry but the row still spans
    // columns 0..=1, so nothing else is determined yet.
    echelon.absorb_unit(2, &pack::<Gf8B>(&[x[2]]));
    assert_eq!(echelon.newly_recovered_columns(), &[2]);
    assert_eq!(echelon.recovered_value(0), None);
    // Recovering column 1 collapses the row to a unit on 0.
    echelon.absorb_unit(1, &pack::<Gf8B>(&[x[1]]));
    assert_eq!(echelon.newly_recovered_columns(), &[1, 0]);
    assert_eq!(
        echelon.recovered_value(0).unwrap(),
        &pack::<Gf8B>(&[x[0]])[..]
    );
    assert_eq!(
        echelon.recovered_value(1).unwrap(),
        &pack::<Gf8B>(&[x[1]])[..]
    );
    assert_eq!(
        echelon.recovered_value(2).unwrap(),
        &pack::<Gf8B>(&[x[2]])[..]
    );
}

#[test]
fn wider_field_matches_ple_solution() {
    // The two-byte field through the same streaming interface.
    type F = Gf16;
    type FElem = <F as Field>::Elem;
    let n = 5;
    let sym = 2;
    let x: Vec<Vec<FElem>> = (0..n).map(|v| elems::<F>(sym, 0xF1E0 + v as u64)).collect();
    let mut echelon = Echelon::<F>::new(n, sym * F::BYTES, true).unwrap();
    let mut a = Matrix::<F>::zeros(n, n).unwrap();
    let mut b = Matrix::<F>::zeros(n, sym).unwrap();
    for (i, packed) in invertible_coeffs::<F>(n, 0xF1E1)
        .into_iter()
        .map(|row| pack::<F>(&row))
        .enumerate()
    {
        let coefficients: Vec<FElem> = packed
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| F::decode(chunk))
            .collect();
        let mut rhs = vec![FElem::ZERO; sym];
        for (j, &c) in coefficients.iter().enumerate() {
            a.set(i, j, c);
            for s in 0..sym {
                rhs[s] = rhs[s].add(c.mul(x[j][s]));
            }
        }
        echelon.absorb(&packed, &pack::<F>(&rhs));
        for (s, &v) in rhs.iter().enumerate() {
            b.set(i, s, v);
        }
    }
    assert!(echelon.is_complete());
    let ple = Ple::decompose(a, &mut PleScratch::new());
    let mut out = Matrix::<F>::zeros(n, sym).unwrap();
    ple.solve_into(&b, &mut out, &mut SolveScratch::new())
        .unwrap();
    for v in 0..n {
        for s in 0..sym {
            assert_eq!(out.get(v, s), x[v][s], "variable {v}, symbol {s}");
        }
        assert_eq!(
            echelon.recovered_value(v).unwrap(),
            out.row(v),
            "recovered variable {v}"
        );
    }
}

#[test]
fn advance_prefix_zero_is_a_noop() {
    let mut echelon = Echelon::<Gf8B>::new(4, 1, true).unwrap();
    echelon.absorb(&[1, 0, 0, 0], &[3]);
    echelon.absorb(&[0, 0, 1, 0], &[5]);
    let pivots: Vec<usize> = echelon.pivot_columns().collect();
    echelon.advance_prefix(0);
    assert_eq!(echelon.rank(), 2);
    assert_eq!(echelon.pivot_columns().collect::<Vec<_>>(), pivots);
}

#[test]
fn advancing_prefix_reindexes_surviving_rows() {
    for reduced in [false, true] {
        let mut echelon = Echelon::<Gf8B>::new(5, 1, reduced).unwrap();
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
                row.mul_coefficients_add(&mut coefficients, Gf8B::decode(&[1]));
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

#[test]
fn advance_then_dense_absorb_matches_ple() {
    // Two unit pivots die in the prefix advance, leaving a nonzero ring
    // offset; the dense rows absorbed afterwards span the wrap point, so
    // reduction and retained-row reads cross from the ring end to its start.
    let (n, off) = (8usize, 4usize);
    let x = elems::<Gf8B>(n, 0x0BAD);
    let mut echelon = Echelon::<Gf8B>::new(n, 1, true).unwrap();
    echelon.absorb_unit(0, &pack::<Gf8B>(&[x[0]]));
    echelon.absorb_unit(1, &pack::<Gf8B>(&[x[1]]));
    echelon.advance_prefix(off);
    assert_eq!(echelon.rank(), 0);

    // A full-rank system posed in logical coordinates: logical `l` is
    // absolute `(l + off) % n`.
    let absolute: Vec<Vec<u8>> = invertible_coeffs::<Gf8B>(n, 0x0BE4)
        .iter()
        .map(|row| pack::<Gf8B>(row))
        .collect();
    let to_logical = |abs: &[u8]| {
        let mut logical = vec![0u8; n];
        for l in 0..n {
            logical[l] = abs[(l + off) % n];
        }
        logical
    };
    for abs_row in &absolute {
        let logical = to_logical(abs_row);
        let coefficients: Vec<E> = logical.iter().map(|&b| Gf8B::decode(&[b])).collect();
        let mut rhs = E::ZERO;
        for l in 0..n {
            rhs = rhs.add(coefficients[l].mul(x[(l + off) % n]));
        }
        assert!(matches!(
            echelon.absorb(&logical, &pack::<Gf8B>(&[rhs])),
            Innovation::Innovative { .. }
        ));
    }
    assert!(echelon.is_complete());

    // Every retained row satisfies its equation at the true solution, and
    // its packed slices agree with the gathered output row.
    for row in echelon.retained_rows() {
        let pivot = row.pivot();
        let (first, second) = row.coefficient_slices();
        let packed: Vec<u8> = first.iter().chain(second).copied().collect();
        let mut gathered = vec![0u8; n];
        row.mul_coefficients_add(&mut gathered, E::ONE);
        assert!(gathered[..pivot].iter().all(|&b| b == 0));
        let last = gathered
            .iter()
            .rposition(|&b| b != 0)
            .expect("retained row has a pivot");
        assert_eq!(packed, gathered[pivot..=last], "slices match gathered row");
        let mut check = E::ZERO;
        for l in 0..n {
            check = check.add(Gf8B::decode(&gathered[l..=l]).mul(x[(l + off) % n]));
        }
        assert_eq!(
            pack::<Gf8B>(&[check]),
            row.rhs(),
            "row satisfies the system"
        );
    }

    // The recovered payloads match a batch Ple over the absolute system.
    let mut a = Matrix::<Gf8B>::zeros(n, n).unwrap();
    let mut b = Matrix::<Gf8B>::zeros(n, 1).unwrap();
    for (i, abs_row) in absolute.iter().enumerate() {
        let mut rhs = E::ZERO;
        for (j, &byte) in abs_row.iter().enumerate() {
            let coef = Gf8B::decode(&[byte]);
            a.set(i, j, coef);
            rhs = rhs.add(coef.mul(x[j]));
        }
        b.set(i, 0, rhs);
    }
    let ple = Ple::decompose(a, &mut PleScratch::new());
    let mut out = Matrix::<Gf8B>::zeros(n, 1).unwrap();
    ple.solve_into(&b, &mut out, &mut SolveScratch::new())
        .unwrap();
    for l in 0..n {
        assert_eq!(
            echelon.recovered_value(l).unwrap(),
            out.row((l + off) % n),
            "recovered logical variable {l}"
        );
    }
}

#[test]
fn cyclic_wrap_second_slice_matches_stored_halves() {
    // A retained row created after a prefix advance stores its span
    // across the ring end: reducing a later row against it drives
    // `mul_add_cyclic` into its second slice, and the gathered output
    // still reproduces the stored halves.
    let n = 10;
    let mut echelon = Echelon::<Gf8B>::new(n, 1, false).unwrap();
    echelon.absorb_unit(0, &[0x11]);
    echelon.absorb_unit(1, &[0x22]);
    echelon.advance_prefix(6);
    // Logical pivot-0 row spanning logical 0..=5 (physical 6..=9 then
    // 0..=1): the stored span wraps, so a later reduction against it
    // exercises the cyclic second slice.
    let mut wide = vec![E::ZERO; n];
    wide[0] = E::ONE;
    wide[5] = Gf8B::decode(&[0x09]);
    assert!(matches!(
        echelon.absorb(&pack::<Gf8B>(&wide), &[0x31]),
        Innovation::Innovative { pivot: 0 }
    ));
    // A row pivoting at 1 with a nonzero entry at 0 reduces against the
    // wrapped pivot-0 row through the cyclic path.
    let mut next = vec![E::ZERO; n];
    next[0] = Gf8B::decode(&[0x04]);
    next[1] = E::ONE;
    assert!(matches!(
        echelon.absorb(&pack::<Gf8B>(&next), &[0x44]),
        Innovation::Innovative { pivot: 1 }
    ));
    // Both retained rows satisfy their equations at the true solution
    // x = [1, 2, 0, ...]: row0 gives rhs 1 + 0, row1 gives 4*1 + 2.
    let x: Vec<E> = (0..n).map(|i| Gf8B::decode(&[(i + 1) as u8])).collect();
    let _ = x;
    for row in echelon.retained_rows() {
        let (first, second) = row.coefficient_slices();
        // The pivot-0 row wraps; the pivot-1 row need not.
        if row.pivot() == 0 {
            assert!(!second.is_empty(), "pivot-0 storage must wrap");
        }
        let mut gathered = vec![0u8; n];
        row.mul_coefficients_add(&mut gathered, E::ONE);
        let packed: Vec<u8> = first.iter().chain(second).copied().collect();
        let pivot = row.pivot();
        assert!(gathered[..pivot].iter().all(|&b| b == 0));
        let last = gathered.iter().rposition(|&b| b != 0).unwrap();
        assert_eq!(packed, gathered[pivot..=last], "pivot {}", row.pivot());
    }
}

#[test]
fn wrapped_gather_matches_stored_slices() {
    // Advancing the prefix moves the ring offset; a later retained row
    // spanning the wrap point stores coefficients on both sides, and the
    // gather into an output row reproduces exactly the stored slices.
    let n = 12;
    let mut echelon = Echelon::<Gf8B>::new(n, 1, false).unwrap();
    echelon.absorb_unit(0, &[0x11]);
    echelon.absorb_unit(1, &[0x22]);
    echelon.advance_prefix(7);
    // Logical row with support straddling the wrap: nonzero at logical 0
    // (physical 7) and at logical 7..=11 (physical 2..=6).
    let mut coefficients = vec![E::ZERO; n];
    coefficients[0] = Gf8B::decode(&[0x03]);
    coefficients[7] = Gf8B::decode(&[0x05]);
    coefficients[11] = Gf8B::decode(&[0x07]);
    let rhs = Gf8B::decode(&[0x2A]);
    assert!(matches!(
        echelon.absorb(&pack::<Gf8B>(&coefficients), &[0x2A]),
        Innovation::Innovative { .. }
    ));
    let _ = rhs;
    let row = echelon
        .retained_rows()
        .find(|row| row.pivot() == 0)
        .expect("pivot 0 retained");
    let (first, second) = row.coefficient_slices();
    assert!(!second.is_empty(), "storage must wrap the ring end");
    // A nonzero factor exercises the kernel gather on both slices.
    let factor = Gf8B::decode(&[0x1B]);
    let mut out = vec![0u8; n];
    row.mul_coefficients_add(&mut out, factor);
    let packed: Vec<u8> = first.iter().chain(second).copied().collect();
    let pivot = row.pivot();
    // Scaling the stored slices by the factor yields the gathered row
    // between the pivot and the last nonzero entry.
    let mut scaled = vec![0u8; packed.len()];
    for (i, chunk) in packed.as_chunks::<1>().0.iter().enumerate() {
        let v = Gf8B::decode(chunk).mul(factor);
        Gf8B::encode(&mut scaled[i..=i], v);
    }
    let last = out.iter().rposition(|&b| b != 0).unwrap();
    assert_eq!(scaled, out[pivot..=last]);
}

#[test]
fn wide_rows_match_ple() {
    // Seventy columns take the out-of-line row-kernel path in both the
    // forward reduction and the right-hand-side gather.
    for reduced in [false, true] {
        let (cols, rhs_cols) = (70usize, 2usize);
        let rows: Vec<Vec<E>> = (0..cols)
            .map(|i| elems::<Gf8B>(cols, 0xB700 + i as u64))
            .collect();
        let mut echelon = Echelon::<Gf8B>::new(cols, rhs_cols, reduced).unwrap();
        let zero_rhs = vec![0u8; rhs_cols];
        for row in &rows {
            echelon.absorb(&pack::<Gf8B>(row), &zero_rhs);
        }
        let mut m = Matrix::<Gf8B>::zeros(rows.len(), cols).unwrap();
        for (i, row) in rows.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                m.set(i, j, e);
            }
        }
        let ple = Ple::decompose(m, &mut PleScratch::new());
        assert_eq!(echelon.rank(), ple.rank(), "wide rank");
        assert_eq!(
            echelon.pivot_columns().collect::<Vec<_>>(),
            ple.col_rank_profile(),
            "wide pivots"
        );
    }
}

#[test]
fn wide_forward_absorb_matches_ple_rank() {
    // Same width without reduction: absorb unit rows first so the pivot
    // set is non-empty, then a dense reduced-mode-style row whose
    // reduction hits both the span-one clearing and the direct kernel
    // update. Reduced mode is off here, so the substitution loop is
    // skipped; the direct update in the forward loop is what runs.
    let cols = 68usize;
    let mut echelon = Echelon::<Gf8B>::new(cols, 2, false).unwrap();
    for c in [0usize, 2, 4] {
        let mut unit = vec![E::ZERO; cols];
        unit[c] = E::ONE;
        echelon.absorb(&pack::<Gf8B>(&unit), &[0x11, 0x22]);
    }
    // A dense row with nonzero entries at the pivoted columns 0, 2, 4:
    // each reduces against a span-one pivot (direct kernel update) and
    // the row stays innovative at pivot 1.
    let row: Vec<E> = (0..cols)
        .map(|j| {
            if j <= 4 {
                Gf8B::decode(&[0x01 + j as u8])
            } else {
                E::ZERO
            }
        })
        .collect();
    assert!(matches!(
        echelon.absorb(&pack::<Gf8B>(&row), &[0x33, 0x44]),
        Innovation::Innovative { pivot: 1 }
    ));
    assert_eq!(echelon.rank(), 4);
    // The oracle over the same four rows agrees on rank and pivots.
    let mut all: Vec<Vec<E>> = Vec::new();
    for c in [0usize, 2, 4] {
        let mut unit = vec![E::ZERO; cols];
        unit[c] = E::ONE;
        all.push(unit);
    }
    all.push(row);
    let mut m = Matrix::<Gf8B>::zeros(all.len(), cols).unwrap();
    for (i, r) in all.iter().enumerate() {
        for (j, &e) in r.iter().enumerate() {
            m.set(i, j, e);
        }
    }
    let ple = Ple::decompose(m, &mut PleScratch::new());
    assert_eq!(echelon.rank(), ple.rank());
    assert_eq!(
        echelon.pivot_columns().collect::<Vec<_>>(),
        ple.col_rank_profile()
    );
}

#[test]
fn wide_reduced_absorb_matches_ple() {
    // Sixty-eight columns take the out-of-line row-kernel path in both the
    // forward reduction and the recovered-column substitution, in reduced
    // mode where recovered units propagate.
    let (n, sym) = (68usize, 2usize);
    let x: Vec<Vec<E>> = (0..n)
        .map(|v| elems::<Gf8B>(sym, 0xE100 + v as u64))
        .collect();
    let coefficients = invertible_with_tail(n, 0xE101);
    let mut echelon = Echelon::<Gf8B>::new(n, sym, true).unwrap();
    let mut a = Matrix::<Gf8B>::zeros(n, n).unwrap();
    let mut b = Matrix::<Gf8B>::zeros(n, sym).unwrap();
    for (i, row) in coefficients.iter().enumerate() {
        let mut rhs = vec![E::ZERO; sym];
        for (j, &c) in row.iter().enumerate() {
            a.set(i, j, c);
            for s in 0..sym {
                rhs[s] = rhs[s].add(c.mul(x[j][s]));
            }
        }
        assert!(matches!(
            echelon.absorb(&pack::<Gf8B>(row), &pack::<Gf8B>(&rhs)),
            Innovation::Innovative { .. }
        ));
        for (s, &v) in rhs.iter().enumerate() {
            b.set(i, s, v);
        }
    }
    assert!(echelon.is_complete());
    assert_eq!(echelon.rank(), n);
    let ple = Ple::decompose(a, &mut PleScratch::new());
    assert_eq!(ple.rank(), n);
    assert_eq!(
        echelon.pivot_columns().collect::<Vec<_>>(),
        ple.col_rank_profile()
    );
    let mut out = Matrix::<Gf8B>::zeros(n, sym).unwrap();
    ple.solve_into(&b, &mut out, &mut SolveScratch::new())
        .unwrap();
    for v in 0..n {
        assert_eq!(out.row(v).to_vec(), pack::<Gf8B>(&x[v]), "variable {v}");
        assert_eq!(
            echelon.recovered_value(v).unwrap(),
            out.row(v),
            "recovered variable {v}"
        );
    }
}

#[test]
fn wide_reduced_substitution_drives_the_kernel_directly() {
    // Reduced mode past sixty-four columns: recover two unit columns,
    // absorb a dense row pivoting at 0 with a nonzero entry at a column
    // that is recovered but has NO pivot. The substitution loop then
    // takes the `None` early-continue (the column is known but carries
    // no equation), and the direct kernel update still applies to the
    // other recovered column.
    let cols = 68usize;
    let mut echelon = Echelon::<Gf8B>::new(cols, 2, true).unwrap();
    let x0 = Gf8B::decode(&[0x17]);
    let x5 = Gf8B::decode(&[0x2B]);
    echelon.absorb_unit(5, &pack::<Gf8B>(&[x5, x5]));
    // Mark column 7 recovered without a pivot: absorb a unit there, then
    // advance it away is impossible (advance drops pivots); instead use
    // a dense row that recovers 7 first, then a second row pivoting at 0
    // with nonzero entries at both 5 (pivot) and 7 (pivot).
    let mut first = vec![E::ZERO; cols];
    first[7] = E::ONE;
    echelon.absorb(&pack::<Gf8B>(&first), &pack::<Gf8B>(&[x0, x0]));
    assert_eq!(
        echelon.recovered_value(7).unwrap(),
        &pack::<Gf8B>(&[x0, x0])[..]
    );
    // Row pivoting at 0 with nonzero entries at recovered columns 5 and
    // 7: both substitute through the kernel path.
    let mut row = vec![E::ZERO; cols];
    row[0] = E::ONE;
    row[5] = Gf8B::decode(&[0x03]);
    row[7] = Gf8B::decode(&[0x05]);
    let rhs0 = x0
        .add(Gf8B::decode(&[0x03]).mul(x5))
        .add(Gf8B::decode(&[0x05]).mul(x0));
    assert!(matches!(
        echelon.absorb(&pack::<Gf8B>(&row), &pack::<Gf8B>(&[rhs0, rhs0])),
        Innovation::Innovative { pivot: 0 }
    ));
    assert_eq!(
        echelon.recovered_value(0).unwrap(),
        &pack::<Gf8B>(&[x0, x0])[..]
    );
    assert_eq!(
        echelon.recovered_value(5).unwrap(),
        &pack::<Gf8B>(&[x5, x5])[..]
    );
}
