//! The dense domain against its contracts: layout invariants as properties,
//! permutations, index-only row exchange, and geometry validation.

mod common;

use common::{draw, noise, sample_dims};
use fgf::field::Elem;
use fgf::{FanPaar32, Gf8, Gf16, Gf64};
use gfm::dense::layout::{ALIGN, pitch_for};
use gfm::{GeometryError, Matrix, Perm};

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
                    F::write(&mut bytes, v);
                    assert_eq!(&bytes[..], &expected[c * F::BYTES..(c + 1) * F::BYTES]);
                }
            }
        }
    }
    check::<Gf8>();
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
    let mut m = Matrix::<Gf8>::zeros(13, 17).unwrap();
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
fn apply_row_perm_matches_permutation_of_indices() {
    let rows = 19;
    let mut m = Matrix::<Gf8>::zeros(rows, 23).unwrap();
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
fn views_see_the_same_rows() {
    let mut m = Matrix::<Gf8>::zeros(21, 15).unwrap();
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
fn mutable_view_edits_the_matrix() {
    let mut m = Matrix::<Gf8>::zeros(9, 7).unwrap();
    {
        let mut v = m.as_view_mut();
        v.set(3, 4, fgf::gf8::Elem(0xAB));
        v.row_mut(5).fill(0x11);
        v.swap_rows(3, 5);
    }
    assert_eq!(m.row(3), &[0x11u8; 7][..]);
    assert_eq!(m.row(5)[4], 0xAB);
    assert!(m.row(5)[..4].iter().all(|&b| b == 0));
    assert!(m.row(5)[5..].iter().all(|&b| b == 0));
    assert_eq!(m.get(5, 4), fgf::gf8::Elem(0xAB));
}

#[test]
fn geometry_errors_are_exact_and_state_preserving() {
    // `rows * pitch` overflow.
    let err = Matrix::<Gf8>::zeros(usize::MAX / 16, 100).unwrap_err();
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
    let err = Matrix::<Gf8>::from_rows(2, 3, &[0u8; 5]).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (2, 3),
            rhs: (5, 1),
        }
    );
    // A mismatched permutation leaves the matrix untouched.
    let mut m = Matrix::<Gf8>::zeros(4, 4).unwrap();
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

/// The physical half of the contract, behind `internals`: alignment, pitch,
/// zero padding, and index-only row exchange.
#[cfg(feature = "internals")]
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
        check_layout::<Gf8>();
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
        let mut m = Matrix::<Gf8>::zeros(rows, 40).unwrap();
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
