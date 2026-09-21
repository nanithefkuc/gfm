//! The dense domain against its contracts: layout invariants as properties,
//! index-only row exchange and compaction, borrow views, geometry
//! validation, permutations, and the bounded compact matrix against the
//! general decomposition.

mod common;
mod oracles;

use common::{all_images, draw, noise, perm_from_image, sample_dims};
use fgf::field::{Elem, Field};
use fgf::{FanPaar32, Gf8B, Gf16, Gf64};
use gfm::dense::layout::{ALIGN, pitch_for};
use gfm::internals::MatrixInternals;
use gfm::{GeometryError, Matrix, Perm, Ple, PleScratch, SmallMatrix, SolveScratch};
use oracles::{crate_matrix, naive_mul, naive_noise, naive_of};

/// Fills every row with deterministic noise, one seed per row.
fn fill_noise<F: fgf::FieldKernels>(m: &mut Matrix<F>, seed: u64) {
    for r in 0..m.rows() {
        let live = m.row(r).len();
        m.row_mut(r).copy_from_slice(&noise(live, seed + r as u64));
    }
}

/// A recorded transposition list: step `i` swaps with a deterministic draw
/// in `i..n`.
fn random_perm(n: usize, seed: u64) -> Perm {
    let mut p = Perm::identity(n);
    let mut state = seed | 1;
    for i in 0..n {
        p.record_swap(i, i + draw(&mut state, n - i));
    }
    p
}

/// A square byte matrix of order `K` for the compact-matrix checks; the
/// deficient form zeroes the last row and column.
fn small_source<const K: usize>(seed: u64, deficient: bool) -> Matrix<Gf8B> {
    let mut state = seed | 1;
    let mut matrix = Matrix::<Gf8B>::zeros(K, K).unwrap();
    for row in 0..K {
        for col in 0..K {
            let value = if deficient && (row + 1 == K || col + 1 == K) {
                <Gf8B as Field>::Elem::ZERO
            } else {
                Gf8B::decode(&[draw(&mut state, 256) as u8])
            };
            matrix.set(row, col, value);
        }
    }
    matrix
}

fn check_rank<const K: usize>() {
    for deficient in [false, true] {
        let matrix = small_source::<K>(0x5A11 ^ K as u64, deficient);
        let small = SmallMatrix::<Gf8B, K>::from_matrix(&matrix);
        let expected = Ple::decompose(matrix, &mut PleScratch::new()).rank();
        assert_eq!(small.rank(), expected, "rank mismatch at K={K}");
    }
}

fn check_solve<const K: usize>() {
    let mut state = 0x501E ^ K as u64;
    let mut matrix = Matrix::<Gf16>::zeros(K, K).unwrap();
    for row in 0..K {
        matrix.set(row, row, <Gf16 as Field>::Elem::ONE);
        for col in (row + 1)..K {
            matrix.set(
                row,
                col,
                Gf16::decode(&(draw(&mut state, u16::MAX as usize) as u16).to_le_bytes()),
            );
        }
    }
    let rhs_bytes = noise(K * 3 * <Gf16 as Field>::BYTES, 0xB001 ^ K as u64);
    let rhs = Matrix::<Gf16>::from_rows(K, 3, &rhs_bytes).unwrap();
    let small = SmallMatrix::<Gf16, K>::from_matrix(&matrix);
    let mut small_out = Matrix::<Gf16>::zeros(K, 3).unwrap();
    small.solve_into(&rhs, &mut small_out).unwrap();
    let ple = Ple::decompose(matrix, &mut PleScratch::new());
    let mut ple_out = Matrix::<Gf16>::zeros(K, 3).unwrap();
    ple.solve_into(&rhs, &mut ple_out, &mut SolveScratch::new())
        .unwrap();
    assert_eq!(small_out, ple_out, "solution mismatch at K={K}");
}

macro_rules! every_order {
    ($check:ident) => {
        $check::<0>();
        $check::<1>();
        $check::<2>();
        $check::<3>();
        $check::<4>();
        $check::<5>();
        $check::<6>();
        $check::<7>();
        $check::<8>();
        $check::<9>();
        $check::<10>();
        $check::<11>();
        $check::<12>();
        $check::<13>();
        $check::<14>();
        $check::<15>();
        $check::<16>();
        $check::<17>();
        $check::<18>();
        $check::<19>();
        $check::<20>();
        $check::<21>();
        $check::<22>();
        $check::<23>();
        $check::<24>();
        $check::<25>();
        $check::<26>();
        $check::<27>();
        $check::<28>();
        $check::<29>();
        $check::<30>();
        $check::<31>();
        $check::<32>();
        $check::<33>();
        $check::<34>();
        $check::<35>();
        $check::<36>();
        $check::<37>();
        $check::<38>();
        $check::<39>();
        $check::<40>();
        $check::<41>();
        $check::<42>();
        $check::<43>();
        $check::<44>();
        $check::<45>();
        $check::<46>();
        $check::<47>();
        $check::<48>();
        $check::<49>();
        $check::<50>();
        $check::<51>();
        $check::<52>();
        $check::<53>();
        $check::<54>();
        $check::<55>();
        $check::<56>();
        $check::<57>();
        $check::<58>();
        $check::<59>();
        $check::<60>();
        $check::<61>();
        $check::<62>();
        $check::<63>();
        $check::<64>();
    };
}

#[test]
fn constructors_and_accessors() {
    fn check<F: fgf::FieldKernels>() {
        for (rows, cols) in sample_dims(0xC0FFEE) {
            let mut m = Matrix::<F>::zeros(rows, cols).unwrap();
            assert_eq!(m.rows(), rows);
            assert_eq!(m.cols(), cols);
            assert_eq!(m.is_square(), rows == cols);
            for r in 0..rows {
                assert_eq!(m.row(r).len(), cols * F::BYTES);
                for c in 0..cols {
                    assert_eq!(m.get(r, c), F::Elem::ZERO);
                }
            }
            fill_noise(&mut m, 0xBEEF);
            for r in 0..rows {
                let expected = noise(cols * F::BYTES, 0xBEEF + r as u64);
                assert_eq!(m.row(r), &expected[..]);
                for c in 0..cols {
                    let v = m.get(r, c);
                    let mut bytes = vec![0u8; F::BYTES];
                    F::encode(&mut bytes, v);
                    assert_eq!(&bytes[..], &expected[c * F::BYTES..(c + 1) * F::BYTES]);
                }
            }
        }
    }
    check::<Gf8B>();
    check::<Gf16>();
    check::<Gf64>();
    check::<FanPaar32>();
}

#[test]
fn identity_has_unit_diagonal() {
    for n in [0, 1, 2, 7, 33] {
        let m = Matrix::<Gf16>::identity(n).unwrap();
        for r in 0..n {
            for c in 0..n {
                let expected = if r == c {
                    fgf::gf16::Elem::ONE
                } else {
                    fgf::gf16::Elem::ZERO
                };
                assert_eq!(m.get(r, c), expected, "({r}, {c})");
            }
        }
    }
}

#[test]
fn from_rows_round_trips() {
    let (rows, cols) = (5, 9);
    let data = noise(rows * cols * 2, 0xF00D);
    let m = Matrix::<Gf16>::from_rows(rows, cols, &data).unwrap();
    for r in 0..rows {
        assert_eq!(m.row(r), &data[r * cols * 2..(r + 1) * cols * 2]);
    }
}

#[test]
fn swap_rows_exchanges_logical_rows() {
    let mut m = Matrix::<Gf8B>::zeros(13, 17).unwrap();
    fill_noise(&mut m, 0xAAAA);
    let (a, b) = (3usize, 11usize);
    let before_a = m.row(a).to_vec();
    let before_b = m.row(b).to_vec();
    m.swap_rows(a, b);
    assert_eq!(m.row(a), &before_b[..]);
    assert_eq!(m.row(b), &before_a[..]);
    m.swap_rows(a, b);
    assert_eq!(m.row(a), &before_a[..]);
    m.swap_rows(a, a);
    assert_eq!(m.row(a), &before_a[..]);
}

#[test]
fn self_swaps_leave_the_matrix_unchanged() {
    let mut m = Matrix::<Gf8B>::zeros(4, 5).unwrap();
    for r in 0..4 {
        m.row_mut(r).copy_from_slice(&noise(5, 0xC303 + r as u64));
    }
    let snapshot = m.clone();
    // `swap_rows` with equal indices is an index no-op.
    m.swap_rows(2, 2);
    assert_eq!(m, snapshot);
    // Compaction after only self-swaps still materializes the same content.
    m.compact_rows();
    assert_eq!(m, snapshot);
}

#[test]
fn apply_row_perm_matches_permutation_of_indices() {
    let rows = 19;
    let mut m = Matrix::<Gf8B>::zeros(rows, 23).unwrap();
    fill_noise(&mut m, 0x1234);
    let p = random_perm(rows, 0x5678);
    // The logical content order after applying `p` equals `p` applied to the
    // row indices: logical row `x` is the old row `image[x]`.
    let mut image: Vec<usize> = (0..rows).collect();
    p.apply(&mut image);
    let before: Vec<Vec<u8>> = (0..rows).map(|r| m.row(r).to_vec()).collect();
    m.apply_row_perm(&p).unwrap();
    for (x, &src) in image.iter().enumerate() {
        assert_eq!(m.row(x), &before[src][..], "row {x}");
    }
}

#[test]
fn compact_rows_preserves_logical_content() {
    // Exhaustive over every permutation of up to 4 rows, then a spread of
    // larger ones: regression coverage for the compact path's cycle walking.
    for n in 1..=4usize {
        for image in common::all_images(n) {
            let mut m = Matrix::<Gf16>::zeros(n, 10).unwrap();
            fill_noise(&mut m, 0x77);
            let before: Vec<Vec<u8>> = (0..n).map(|r| m.row(r).to_vec()).collect();
            m.apply_row_perm(&common::perm_from_image(&image)).unwrap();
            let logical: Vec<Vec<u8>> = (0..n).map(|r| m.row(r).to_vec()).collect();
            for (x, &src) in image.iter().enumerate() {
                assert_eq!(m.row(x), &before[src][..], "permuted row {x}");
            }
            m.compact_rows();
            for (r, expected) in logical.iter().enumerate() {
                assert_eq!(m.row(r), &expected[..], "compacted row {r}");
            }
        }
    }
    for seed in 0..20u64 {
        let rows = 64;
        let mut m = Matrix::<Gf16>::zeros(rows, 10).unwrap();
        fill_noise(&mut m, 0x77);
        let p = random_perm(rows, 0x8800 + seed);
        m.apply_row_perm(&p).unwrap();
        let logical: Vec<Vec<u8>> = (0..rows).map(|r| m.row(r).to_vec()).collect();
        m.compact_rows();
        for (r, expected) in logical.iter().enumerate() {
            assert_eq!(m.row(r), &expected[..], "row {r}");
        }
    }
}

#[test]
fn compact_rows_materializes_logical_content() {
    // A row permutation followed by compaction: logical content is
    // unchanged and the map is the identity afterwards.
    let (rows, cols) = (6usize, 5usize);
    let mut m = Matrix::<Gf8B>::zeros(rows, cols).unwrap();
    for r in 0..rows {
        m.row_mut(r)
            .copy_from_slice(&noise(cols, 0xC301 + r as u64));
    }
    let mut permuted = m.clone();
    permuted.swap_rows(0, 5);
    permuted.swap_rows(1, 3);
    let before: Vec<Vec<u8>> = (0..rows).map(|r| permuted.row(r).to_vec()).collect();
    permuted.compact_rows();
    for (r, row) in before.iter().enumerate() {
        assert_eq!(permuted.row(r), &row[..], "row {r}");
    }
    // Swapping through the compacted map still exchanges logical rows.
    permuted.swap_rows(0, 5);
    assert_eq!(permuted.row(0), &before[5][..]);
    assert_eq!(permuted.row(5), &before[0][..]);
}

#[test]
fn compact_rows_on_identity_is_a_noop() {
    let (rows, cols) = (4usize, 7usize);
    let mut m = Matrix::<Gf8B>::zeros(rows, cols).unwrap();
    for r in 0..rows {
        m.row_mut(r)
            .copy_from_slice(&noise(cols, 0xC302 + r as u64));
    }
    let snapshot = m.clone();
    m.compact_rows();
    assert_eq!(m, snapshot);
}

#[test]
fn views_see_the_same_rows() {
    let mut m = Matrix::<Gf8B>::zeros(21, 15).unwrap();
    fill_noise(&mut m, 0x99);
    m.swap_rows(2, 19);
    let view = m.as_view();
    assert_eq!(view.rows(), m.rows());
    assert_eq!(view.cols(), m.cols());
    assert_eq!(view.pitch(), m.pitch());
    for r in 0..m.rows() {
        assert_eq!(view.row(r), m.row(r));
    }
    // Split at every boundary, including empty halves.
    for at in 0..=m.rows() {
        let (top, bot) = view.split_rows(at).unwrap();
        assert_eq!(top.rows(), at);
        assert_eq!(bot.rows(), m.rows() - at);
        for r in 0..at {
            assert_eq!(top.row(r), m.row(r));
        }
        for r in 0..m.rows() - at {
            assert_eq!(bot.row(r), m.row(at + r));
        }
    }
    assert!(view.split_rows(m.rows() + 1).is_none());
}

#[test]
fn view_get_follows_the_row_map() {
    let mut m = Matrix::<Gf16>::zeros(6, 5).unwrap();
    for r in 0..6 {
        m.row_mut(r)
            .copy_from_slice(&noise(5 * 2, 0xB100 + r as u64));
    }
    m.swap_rows(1, 4);
    let view = m.as_view();
    assert_eq!(view.rows(), 6);
    assert_eq!(view.cols(), 5);
    for r in 0..6 {
        for c in 0..5 {
            assert_eq!(view.get(r, c), m.get(r, c), "view element ({r}, {c})");
        }
        assert_eq!(view.row(r), m.row(r), "view row {r}");
    }
    // Swapping back through the owner is visible through a fresh view.
    m.swap_rows(1, 4);
    let view = m.as_view();
    for r in 0..6 {
        for c in 0..5 {
            assert_eq!(view.get(r, c), m.get(r, c));
        }
    }
}

#[test]
fn mutable_view_edits_the_matrix() {
    let mut m = Matrix::<Gf8B>::zeros(9, 7).unwrap();
    {
        let mut v = m.as_view_mut();
        v.set(3, 4, fgf::gf8b::Elem::from_raw(0xAB));
        v.row_mut(5).fill(0x11);
        v.swap_rows(3, 5);
    }
    assert_eq!(m.row(3), &[0x11u8; 7][..]);
    assert_eq!(m.row(5)[4], 0xAB);
    assert!(m.row(5)[..4].iter().all(|&b| b == 0));
    assert!(m.row(5)[5..].iter().all(|&b| b == 0));
    assert_eq!(m.get(5, 4), fgf::gf8b::Elem::from_raw(0xAB));
}

#[test]
fn mutable_view_reports_its_geometry() {
    let mut m = Matrix::<Gf16>::zeros(4, 9).unwrap();
    let pitch = m.pitch();
    {
        let view = m.as_view_mut();
        assert_eq!(view.rows(), 4);
        assert_eq!(view.cols(), 9);
        assert_eq!(view.pitch(), pitch);
    }
    // And element access through the mutable view writes the matrix.
    {
        let mut view = m.as_view_mut();
        view.set(2, 7, Gf16::decode(&[0x11, 0x22]));
        assert_eq!(view.get(2, 7), Gf16::decode(&[0x11, 0x22]));
        assert_eq!(view.row(2)[7 * 2..8 * 2], [0x11, 0x22]);
    }
    assert_eq!(m.get(2, 7), Gf16::decode(&[0x11, 0x22]));
}

#[test]
fn geometry_errors_are_exact_and_state_preserving() {
    // `rows * pitch` overflow.
    let err = Matrix::<Gf8B>::zeros(usize::MAX / 16, 100).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Overflow {
            rows: usize::MAX / 16,
            pitch: 128,
        }
    );
    // `cols * BYTES` overflow.
    let err = Matrix::<Gf16>::zeros(1, usize::MAX).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Overflow {
            rows: usize::MAX,
            pitch: 2,
        }
    );
    // Ragged input.
    let err = Matrix::<Gf16>::from_rows(2, 3, &[0u8; 7]).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Ragged {
            len: 7,
            element_bytes: 2,
        }
    );
    // Wrong element count.
    let err = Matrix::<Gf8B>::from_rows(2, 3, &[0u8; 5]).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (2, 3),
            rhs: (5, 1),
        }
    );
    // A mismatched permutation leaves the matrix untouched.
    let mut m = Matrix::<Gf8B>::zeros(4, 4).unwrap();
    fill_noise(&mut m, 0x42);
    let snapshot = m.clone();
    let err = m.apply_row_perm(&Perm::identity(3)).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (4, 4),
            rhs: (3, 3),
        }
    );
    assert_eq!(m, snapshot);
}

#[test]
fn apply_row_perm_inv_rejects_mismatched_length() {
    let mut m = Matrix::<Gf8B>::zeros(4, 5).unwrap();
    for r in 0..4 {
        m.row_mut(r).copy_from_slice(&noise(5, 0xE100 + r as u64));
    }
    let snapshot = m.clone();
    let err = m.apply_row_perm_inv(&Perm::identity(3)).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (4, 5),
            rhs: (3, 3),
        }
    );
    assert_eq!(m, snapshot, "matrix changed on a rejected permutation");

    // The matching length applies the inverse: apply then apply_inv is the
    // identity on logical content.
    let mut p = Perm::identity(4);
    p.record_swap(0, 2);
    p.record_swap(1, 3);
    m.apply_row_perm(&p).unwrap();
    m.apply_row_perm_inv(&p).unwrap();
    assert_eq!(m, snapshot);
}

#[test]
fn perm_round_trips() {
    for n in [0, 1, 2, 3, 17, 64] {
        let p = random_perm(n, 0xD15E);
        let mut probe: Vec<usize> = (0..n).collect();
        let original = probe.clone();
        p.apply(&mut probe);
        p.apply_inv(&mut probe);
        assert_eq!(probe, original, "round trip at n = {n}");
    }
}

#[test]
fn perm_compose_is_sequential_application() {
    let n = 23;
    let p = random_perm(n, 0x1111);
    let q = random_perm(n, 0x2222);
    let composed = p.compose(&q);
    let mut sequential: Vec<usize> = (0..n).collect();
    p.apply(&mut sequential);
    q.apply(&mut sequential);
    let mut one_shot: Vec<usize> = (0..n).collect();
    composed.apply(&mut one_shot);
    assert_eq!(one_shot, sequential);
}

#[test]
fn perm_compose_is_associative() {
    let n = 15;
    let p = random_perm(n, 0x3333);
    let q = random_perm(n, 0x4444);
    let r = random_perm(n, 0x5555);
    assert_eq!(
        p.compose(&q).compose(&r),
        p.compose(&q.compose(&r)),
        "structural equality of both associations"
    );
}

#[test]
fn perm_parity_matches_inversion_count() {
    for n in [1, 2, 5, 12, 33] {
        let p = random_perm(n, 0x6666 + n as u64);
        let mut image: Vec<usize> = (0..n).collect();
        p.apply(&mut image);
        let mut inversions = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if image[i] > image[j] {
                    inversions += 1;
                }
            }
        }
        assert_eq!(p.parity(), inversions % 2 == 1, "parity at n = {n}");
    }
}

#[test]
fn perm_identity_is_neutral() {
    let n = 9;
    let id = Perm::identity(n);
    assert!(!id.parity());
    let p = random_perm(n, 0x7777);
    assert_eq!(id.compose(&p), p);
    assert_eq!(p.compose(&id), p);
    let mut probe: Vec<usize> = (0..n).collect();
    let original = probe.clone();
    id.apply(&mut probe);
    assert_eq!(probe, original);
}

#[test]
fn perm_image_and_emptiness() {
    assert!(Perm::identity(0).is_empty());
    assert!(!Perm::identity(4).is_empty());
    // Exhaustive over every permutation of up to 4 positions: the recorded
    // image query agrees with the explicit image, through both swap arms.
    for n in 0..=4usize {
        for image in all_images(n) {
            let p = perm_from_image(&image);
            for (position, &expected) in image.iter().enumerate() {
                assert_eq!(
                    p.image_at(position),
                    expected,
                    "image at {position} of {image:?}"
                );
            }
        }
    }
}

#[test]
fn small_rank_agrees_with_ple_at_every_order() {
    every_order!(check_rank);
}

#[test]
fn small_solve_agrees_with_ple_at_every_order() {
    every_order!(check_solve);
}

#[test]
fn small_get_and_set_round_trip() {
    let mut small = SmallMatrix::<Gf8B, 4>::zeros();
    let mut state = 0x5EEDu64;
    for row in 0..4 {
        for col in 0..4 {
            let v = Gf8B::decode(&[draw(&mut state, 256) as u8]);
            small.set(row, col, v);
            assert_eq!(small.get(row, col), v, "element ({row}, {col})");
        }
    }
    // Setting one cell leaves its neighbours alone.
    let (before_right, before_below) = (small.get(1, 1), small.get(2, 1));
    small.set(1, 2, Gf8B::decode(&[0xAB]));
    assert_eq!(small.get(1, 2), Gf8B::decode(&[0xAB]));
    assert_eq!(small.get(1, 1), before_right);
    assert_eq!(small.get(2, 1), before_below);
}

#[test]
fn small_default_is_zero() {
    let small = SmallMatrix::<Gf8B, 3>::default();
    assert_eq!(small.rank(), 0);
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(small.get(row, col), <Gf8B as Field>::Elem::ZERO);
        }
    }
}

#[test]
fn small_rank_pivots_past_a_dead_leading_column() {
    // An anti-diagonal identity: the first pivot search finds nothing in
    // column 0 and must exchange columns before eliminating.
    let mut m = Matrix::<Gf8B>::zeros(4, 4).unwrap();
    for i in 0..4 {
        m.set(i, 3 - i, <Gf8B as Field>::Elem::ONE);
    }
    let small = SmallMatrix::<Gf8B, 4>::from_matrix(&m);
    assert_eq!(small.rank(), 4);
    let expected = Ple::decompose(m, &mut PleScratch::new()).rank();
    assert_eq!(expected, 4);

    // Zeroing a row drops the rank by exactly one.
    let mut m = Matrix::<Gf8B>::zeros(4, 4).unwrap();
    for i in 0..3 {
        m.set(i, 3 - i, <Gf8B as Field>::Elem::ONE);
    }
    let small = SmallMatrix::<Gf8B, 4>::from_matrix(&m);
    assert_eq!(small.rank(), 3);
}

#[test]
fn small_singular_byte_matrix_names_rank_and_order() {
    // Rank 2 of order 4: the last two rows duplicate the first two.
    let top = [noise(2, 0x5101), noise(2, 0x5102)];
    let mut a = Matrix::<Gf8B>::zeros(4, 4).unwrap();
    for (r, row) in top.iter().enumerate() {
        for (c, &byte) in row.iter().enumerate() {
            let v = Gf8B::decode(&[byte]);
            a.set(r, c, v);
            a.set(r + 2, c, v);
        }
    }
    let small = SmallMatrix::<Gf8B, 4>::from_matrix(&a);
    assert_eq!(small.rank(), 2);
    let mut out = Matrix::<Gf8B>::zeros(4, 1).unwrap();
    let rhs = Matrix::<Gf8B>::from_rows(4, 1, &noise(4, 0x5103)).unwrap();
    assert_eq!(
        small.solve_into(&rhs, &mut out),
        Err(gfm::SolveError::Singular { rank: 2, order: 4 })
    );
    // The reported rank agrees with the general decomposition.
    assert_eq!(Ple::decompose(a, &mut PleScratch::new()).rank(), 2);
}

#[test]
fn small_wide_solve_matches_ple_through_swaps_and_scaling() {
    // A permuted diagonal with non-unit entries over the two-byte field:
    // every pivot needs a row exchange and a scale.
    let mut state = 0x1DE0u64;
    let perm: Vec<usize> = {
        let mut p: Vec<usize> = (0..4).collect();
        for i in 0..4 {
            let j = i + draw(&mut state, 4 - i);
            p.swap(i, j);
        }
        p
    };
    let mut a = Matrix::<Gf16>::zeros(4, 4).unwrap();
    for (i, &p) in perm.iter().enumerate() {
        let v = Gf16::decode(&(1 + draw(&mut state, 0xFFFE) as u16).to_le_bytes());
        a.set(i, p, v);
    }
    let rhs = crate_matrix::<Gf16>(&naive_noise::<Gf16>(4, 3, 0x5104));
    let small = SmallMatrix::<Gf16, 4>::from_matrix(&a);
    assert_eq!(small.rank(), 4);
    let mut small_out = Matrix::<Gf16>::zeros(4, 3).unwrap();
    small.solve_into(&rhs, &mut small_out).unwrap();
    let ple = Ple::decompose(a.clone(), &mut PleScratch::new());
    let mut ple_out = Matrix::<Gf16>::zeros(4, 3).unwrap();
    ple.solve_into(&rhs, &mut ple_out, &mut SolveScratch::new())
        .unwrap();
    assert_eq!(small_out, ple_out);
    // The answer satisfies the system by naive multiplication.
    let product = naive_mul::<Gf16>(&naive_of::<Gf16>(&a), &naive_of::<Gf16>(&small_out));
    assert_eq!(product, naive_of::<Gf16>(&rhs));
}

#[test]
fn small_wide_singular_names_rank_and_order() {
    // Rank 1 of order 3 over the two-byte field.
    let mut a = Matrix::<Gf16>::zeros(3, 3).unwrap();
    for c in 0..3 {
        a.set(0, c, Gf16::decode(&[7, 9]));
    }
    for r in 1..3 {
        for c in 0..3 {
            a.set(r, c, a.get(0, c));
        }
    }
    let small = SmallMatrix::<Gf16, 3>::from_matrix(&a);
    assert_eq!(small.rank(), 1);
    let rhs = Matrix::<Gf16>::zeros(3, 2).unwrap();
    let mut out = Matrix::<Gf16>::zeros(3, 2).unwrap();
    assert_eq!(
        small.solve_into(&rhs, &mut out),
        Err(gfm::SolveError::Singular { rank: 1, order: 3 })
    );
}

/// Row byte widths of the shapes the benchmark suite exercises.
const SUITE_ROW_BYTES: &[usize] = &[
    32, 64, 128, 256, 512, 560, 1000, 1024, 1100, 1500, 2048, 3000, 4096, 8192, 16384, 65536,
];

#[test]
fn padding_cost_is_bounded() {
    for &w in SUITE_ROW_BYTES {
        let pitch = pitch_for(w).unwrap();
        // The absolute bound: less than one lane of padding per row.
        assert!(pitch < w + ALIGN, "row bytes {w}: pitch {pitch}");
        // The suite bound: under 5% for rows above 512 bytes (see
        // BENCHMARKS.md for the recorded ratios).
        if w > 512 {
            assert!(
                pitch * 20 < w * 21,
                "row bytes {w}: pitch {pitch} is >= 5% padding",
            );
        }
    }
}

// ── The rectangular product and the triangular solves ────────────────────────

#[test]
fn mul_into_and_mul_add_match_the_naive_product() {
    let a_naive = oracles::naive_noise::<Gf16>(7, 5, 0x1A);
    let b_naive = oracles::naive_noise::<Gf16>(5, 6, 0x2B);
    let expected = oracles::naive_mul::<Gf16>(&a_naive, &b_naive);
    let a = oracles::crate_matrix(&a_naive);
    let b = oracles::crate_matrix(&b_naive);

    let mut out = Matrix::<Gf16>::zeros(7, 6).unwrap();
    gfm::mul_into(&mut out, &a, &b);
    assert_eq!(oracles::naive_of(&out), expected);

    // Accumulation from a zeroed destination is the product; accumulating
    // again doubles it, so `mul_add` folds into the operand's prior value.
    let mut acc = Matrix::<Gf16>::zeros(7, 6).unwrap();
    gfm::mul_add(&mut acc, &a, &b);
    assert_eq!(oracles::naive_of(&acc), expected);
    gfm::mul_add(&mut acc, &a, &b);
    let doubled: oracles::Naive<Gf16> = expected
        .iter()
        .map(|row| row.iter().map(|&v| v.add(v)).collect())
        .collect();
    assert_eq!(oracles::naive_of(&acc), doubled);
}

#[test]
#[should_panic(expected = "matrix multiply shape mismatch")]
fn mul_rejects_mismatched_shapes() {
    let a = Matrix::<Gf16>::zeros(3, 4).unwrap();
    let b = Matrix::<Gf16>::zeros(3, 4).unwrap();
    let mut out = Matrix::<Gf16>::zeros(3, 4).unwrap();
    gfm::mul_into(&mut out, &a, &b);
}

#[test]
fn triangular_solves_round_trip_against_the_product() {
    let (k, s) = (6usize, 3usize);
    let mut state = 0x51u64;
    let byte = |state: &mut u64| Gf8B::decode(&[(1 + draw(state, 255)) as u8]);
    let mut l = Matrix::<Gf8B>::zeros(k, k).unwrap();
    let mut u = Matrix::<Gf8B>::zeros(k, k).unwrap();
    for r in 0..k {
        for c in 0..k {
            let v = byte(&mut state);
            if c < r {
                l.set(r, c, v);
            }
            if c > r {
                u.set(r, c, v);
            }
        }
        u.set(r, r, byte(&mut state));
    }
    let b_naive = oracles::naive_noise::<Gf8B>(k, s, 0x33);
    let b = oracles::crate_matrix(&b_naive);
    let mut x = b.clone();
    gfm::solve_lower_unit_assign(&mut x, &l);
    let mut l_full = l.clone();
    for i in 0..k {
        l_full.set(i, i, <Gf8B as Field>::Elem::ONE);
    }
    let mut prod = Matrix::<Gf8B>::zeros(k, s).unwrap();
    gfm::mul_into(&mut prod, &l_full, &x);
    assert_eq!(oracles::naive_of(&prod), b_naive, "L·X == B");

    let mut y = b;
    gfm::solve_upper_assign(&mut y, &u);
    gfm::mul_into(&mut prod, &u, &y);
    assert_eq!(oracles::naive_of(&prod), b_naive, "U·Y == B");
}

#[test]
#[should_panic(expected = "zero diagonal entry")]
fn upper_solve_rejects_a_zero_diagonal() {
    let mut u = Matrix::<Gf8B>::zeros(2, 2).unwrap();
    u.set(0, 1, <Gf8B as Field>::Elem::ONE);
    let mut b = Matrix::<Gf8B>::zeros(2, 1).unwrap();
    gfm::solve_upper_assign(&mut b, &u);
}

#[test]
#[should_panic(expected = "right-hand side row mismatch")]
fn triangular_solves_reject_a_row_mismatch() {
    let l = Matrix::<Gf8B>::zeros(2, 2).unwrap();
    let mut b = Matrix::<Gf8B>::zeros(3, 1).unwrap();
    gfm::solve_lower_unit_assign(&mut b, &l);
}

#[test]
fn inverse_round_trips_through_the_product() {
    let m_naive = oracles::naive_noise::<Gf8B>(5, 5, 0x77);
    let m = oracles::crate_matrix(&m_naive);
    let ple = Ple::decompose(m.clone(), &mut PleScratch::new());
    assert_eq!(ple.rank(), 5, "seed 0x77 must give an invertible matrix");
    let mut inv = Matrix::<Gf8B>::zeros(5, 5).unwrap();
    ple.inverse_into(&mut inv).unwrap();
    let mut prod = Matrix::<Gf8B>::zeros(5, 5).unwrap();
    gfm::mul_into(&mut prod, &m, &inv);
    let identity = Matrix::<Gf8B>::identity(5).unwrap();
    assert_eq!(oracles::naive_of(&prod), oracles::naive_of(&identity));
}

#[test]
fn redecompose_replaces_the_matrix_in_place() {
    use gfm::internals::PleInternals;

    let a1: Matrix<Gf16> = oracles::crate_matrix(&oracles::naive_noise::<Gf16>(9, 9, 0xA1));
    let mut ple = Ple::decompose(a1, &mut PleScratch::new());
    let a2: Matrix<Gf16> = oracles::crate_matrix(&oracles::naive_noise::<Gf16>(9, 9, 0xB2));
    let a2_rows = a2.rows();
    let fresh = Ple::decompose(a2.clone(), &mut PleScratch::new());
    ple.redecompose_scratch(&mut PleScratch::new(), |lu| {
        for r in 0..a2_rows {
            lu.row_mut(r).copy_from_slice(a2.row(r));
        }
    });
    assert_eq!(ple.rank(), fresh.rank());
    assert_eq!(ple.lu(), fresh.lu());
    assert_eq!(ple.p(), fresh.p());
    assert_eq!(ple.q(), fresh.q());
}

/// The physical half of the contract, behind `internals`: alignment, pitch,
/// zero padding, and index-only row exchange.
mod physical {
    use super::*;
    use common::straddling_dims;
    use fgf::{FanPaar8, FanPaar16, FanPaar64, Gf32};

    fn check_layout<F: fgf::FieldKernels>() {
        let mut dims = sample_dims(0xFEED);
        dims.extend(straddling_dims(F::BYTES, ALIGN));
        for (rows, cols) in dims {
            let mut m = Matrix::<F>::zeros(rows, cols).unwrap();
            let live = cols * F::BYTES;
            assert_eq!(m.base_addr() % ALIGN, 0, "base at ({rows}, {cols})");
            assert_eq!(m.pitch() % ALIGN, 0, "pitch at ({rows}, {cols})");
            assert!(m.pitch() >= live, "pitch covers at ({rows}, {cols})");
            assert_padding_zero(&m);
            fill_noise(&mut m, 0x5EED);
            assert_padding_zero(&m);
            // A sweep of swaps and permutations must not disturb padding.
            let mut state = 0xD00D_u64;
            if rows > 0 {
                for _ in 0..rows * 2 {
                    m.swap_rows(draw(&mut state, rows), draw(&mut state, rows));
                }
            }
            assert_padding_zero(&m);
            let p = random_perm(rows, 0xCACA);
            m.apply_row_perm(&p).unwrap();
            assert_padding_zero(&m);
            m.compact_rows();
            assert_padding_zero(&m);
            for r in 0..rows {
                assert_eq!(m.physical_row_index(r), r, "compact resets the map");
            }
        }
    }

    fn assert_padding_zero<F: fgf::FieldKernels>(m: &Matrix<F>) {
        let live = m.cols() * F::BYTES;
        for r in 0..m.rows() {
            let phys = m.physical_row_index(r);
            let row = &m.pitched_buffer()[phys * m.pitch()..(phys + 1) * m.pitch()];
            assert!(
                row[live..].iter().all(|&b| b == 0),
                "padding nonzero at row {r} of {:?}",
                m,
            );
        }
    }

    #[test]
    fn layout_invariants_hold_for_all_fields() {
        check_layout::<Gf8B>();
        check_layout::<Gf16>();
        check_layout::<Gf32>();
        check_layout::<Gf64>();
        check_layout::<FanPaar8>();
        check_layout::<FanPaar16>();
        check_layout::<FanPaar32>();
        check_layout::<FanPaar64>();
    }

    #[test]
    fn swap_rows_moves_no_data() {
        let rows = 11;
        let mut m = Matrix::<Gf8B>::zeros(rows, 40).unwrap();
        fill_noise(&mut m, 0xBEE0);
        let physical_before = m.pitched_buffer().to_vec();
        let map_before: Vec<usize> = (0..rows).map(|r| m.physical_row_index(r)).collect();
        m.swap_rows(4, 9);
        // Byte-for-byte, the physical buffer is untouched.
        assert_eq!(m.pitched_buffer(), &physical_before[..]);
        // And only the two map entries exchanged places.
        for r in 0..rows {
            let expected = match r {
                4 => map_before[9],
                9 => map_before[4],
                _ => map_before[r],
            };
            assert_eq!(m.physical_row_index(r), expected, "row {r}");
        }
    }
}
