//! The GF(2) domain against its contracts: word-packed layout invariants,
//! bit accessors, masking of out-of-range bits, and index-only row exchange.

mod common;

use common::{draw, noise, sample_dims};
use gfm::{BitMatrix, GeometryError, Perm};

/// `ceil(cols / 64)`.
fn row_words(cols: usize) -> usize {
    cols.div_ceil(64)
}

/// Mask of live bits in the last word of a row (mirrors the crate's own).
fn live_mask(cols: usize) -> u64 {
    if cols.is_multiple_of(64) {
        u64::MAX
    } else {
        (1 << (cols % 64)) - 1
    }
}

/// Deterministic bit content for a row, as live words with clean padding.
fn row_bits(rows: usize, cols: usize, seed: u64) -> Vec<u64> {
    let words = rows * row_words(cols);
    let data = noise(words * 8, seed);
    let mut as_words: Vec<u64> = data
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if row_words(cols) > 0 {
        for w in as_words
            .iter_mut()
            .skip(row_words(cols) - 1)
            .step_by(row_words(cols))
        {
            *w &= live_mask(cols);
        }
    }
    as_words
}

#[test]
fn constructors_and_accessors() {
    for (rows, cols) in sample_dims(0xB17) {
        let m = BitMatrix::zeros(rows, cols).unwrap();
        assert_eq!(m.rows(), rows);
        assert_eq!(m.cols(), cols);
        assert_eq!(m.row_words(), row_words(cols));
        assert_eq!(m.is_square(), rows == cols);
        for r in 0..rows {
            assert_eq!(m.row(r).len(), row_words(cols));
            for c in 0..cols {
                assert!(!m.get(r, c));
            }
        }
        let bits = row_bits(rows, cols, 0xBEEF);
        let mut m = BitMatrix::from_rows(rows, cols, &bits).unwrap();
        for r in 0..rows {
            assert_eq!(
                m.row(r),
                &bits[r * row_words(cols)..(r + 1) * row_words(cols)]
            );
            for c in 0..cols {
                let expected = bits[r * row_words(cols) + c / 64] & (1 << (c % 64)) != 0;
                assert_eq!(m.get(r, c), expected, "({r}, {c})");
            }
        }
        // `set` round-trips every bit and cannot reach padding.
        for r in 0..rows {
            for c in 0..cols {
                m.set(r, c, !m.get(r, c));
            }
        }
        for r in 0..rows {
            let expected: Vec<u64> = bits[r * row_words(cols)..(r + 1) * row_words(cols)]
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let flip = if i == row_words(cols) - 1 {
                        live_mask(cols)
                    } else {
                        u64::MAX
                    };
                    w ^ flip
                })
                .collect();
            assert_eq!(m.row(r), &expected[..]);
        }
    }
}

#[test]
fn identity_has_unit_diagonal() {
    for n in [0, 1, 2, 7, 65] {
        let m = BitMatrix::identity(n).unwrap();
        for r in 0..n {
            for c in 0..n {
                assert_eq!(m.get(r, c), r == c, "({r}, {c})");
            }
        }
    }
}

#[test]
fn from_rows_masks_bits_beyond_cols() {
    // Every bit set in the input, including out-of-range ones.
    let (rows, cols) = (3, 70);
    let data = vec![u64::MAX; rows * row_words(cols)];
    let m = BitMatrix::from_rows(rows, cols, &data).unwrap();
    for r in 0..rows {
        let row = m.row(r);
        assert_eq!(row[0], u64::MAX);
        // 70 = 64 + 6: only the low 6 bits of the second word are live.
        assert_eq!(row[1], (1 << 6) - 1);
    }
}

#[test]
fn swap_rows_exchanges_logical_rows() {
    let mut m = BitMatrix::from_rows(13, 100, &row_bits(13, 100, 0xAAAA)).unwrap();
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
fn permute_and_compact_preserve_logical_content() {
    // Exhaustive over every permutation of up to 4 rows, then a spread of
    // larger ones: regression coverage for the compact path's cycle walking.
    for n in 1..=4usize {
        for image in common::all_images(n) {
            let mut m = BitMatrix::from_rows(n, 70, &row_bits(n, 70, 0x77)).unwrap();
            let before: Vec<Vec<u64>> = (0..n).map(|r| m.row(r).to_vec()).collect();
            m.apply_row_perm(&common::perm_from_image(&image)).unwrap();
            let logical: Vec<Vec<u64>> = (0..n).map(|r| m.row(r).to_vec()).collect();
            for (x, &src) in image.iter().enumerate() {
                assert_eq!(m.row(x), &before[src][..], "permuted row {x}");
            }
            m.compact_rows();
            for (r, expected) in logical.iter().enumerate() {
                assert_eq!(m.row(r), &expected[..], "compacted row {r}");
            }
        }
    }
    let rows = 14;
    let mut m = BitMatrix::from_rows(rows, 70, &row_bits(rows, 70, 0x77)).unwrap();
    let mut p = Perm::identity(rows);
    let mut state = 0x88_u64;
    for i in 0..rows {
        p.record_swap(i, i + draw(&mut state, rows - i));
    }
    let mut image: Vec<usize> = (0..rows).collect();
    p.apply(&mut image);
    let before: Vec<Vec<u64>> = (0..rows).map(|r| m.row(r).to_vec()).collect();
    m.apply_row_perm(&p).unwrap();
    for (x, &src) in image.iter().enumerate() {
        assert_eq!(m.row(x), &before[src][..], "row {x}");
    }
    let logical: Vec<Vec<u64>> = (0..rows).map(|r| m.row(r).to_vec()).collect();
    m.compact_rows();
    for (r, expected) in logical.iter().enumerate() {
        assert_eq!(m.row(r), &expected[..], "row {r}");
    }
}

#[test]
fn geometry_errors_are_exact_and_state_preserving() {
    let err = BitMatrix::zeros(usize::MAX / 4, 100).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Overflow {
            rows: usize::MAX / 4,
            pitch: 8,
        }
    );
    // `cols + 63` overflow.
    let err = BitMatrix::zeros(1, usize::MAX).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Overflow {
            rows: usize::MAX,
            pitch: 64,
        }
    );
    let err = BitMatrix::from_rows(2, 70, &[0u64; 5]).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (2, 70),
            rhs: (5, 1),
        }
    );
    let mut m = BitMatrix::from_rows(4, 40, &row_bits(4, 40, 0x42)).unwrap();
    let snapshot = m.clone();
    let err = m.apply_row_perm(&Perm::identity(3)).unwrap_err();
    assert_eq!(
        err,
        GeometryError::Shape {
            lhs: (4, 40),
            rhs: (3, 3),
        }
    );
    assert_eq!(m, snapshot);
}

/// The physical half of the contract, behind `internals`.
#[cfg(feature = "internals")]
mod physical {
    use super::*;
    use gfm::bits::ALIGN;

    #[test]
    fn layout_invariants_hold() {
        for (rows, cols) in sample_dims(0xFEED) {
            let mut m = BitMatrix::from_rows(rows, cols, &row_bits(rows, cols, 0x5EED)).unwrap();
            assert_eq!(m.base_addr() % ALIGN, 0, "base at ({rows}, {cols})");
            assert_eq!(m.pitch() % ALIGN, 0, "pitch at ({rows}, {cols})");
            assert!(m.pitch() * 8 >= cols, "pitch covers at ({rows}, {cols})",);
            assert_padding_zero(&m);
            let mut state = 0xD00D_u64;
            if rows > 0 {
                for _ in 0..rows * 2 {
                    m.swap_rows(draw(&mut state, rows), draw(&mut state, rows));
                }
            }
            assert_padding_zero(&m);
            let mut p = Perm::identity(rows);
            for i in 0..rows {
                p.record_swap(i, i + draw(&mut state, rows - i));
            }
            m.apply_row_perm(&p).unwrap();
            assert_padding_zero(&m);
            m.compact_rows();
            assert_padding_zero(&m);
            for r in 0..rows {
                assert_eq!(m.physical_row_index(r), r, "compact resets the map");
            }
        }
    }

    fn assert_padding_zero(m: &BitMatrix) {
        let live = m.row_words();
        let mask = live_mask(m.cols());
        for r in 0..m.rows() {
            let phys = m.physical_row_index(r);
            let row = &m.pitched_buffer()[phys * (m.pitch() / 8)..(phys + 1) * (m.pitch() / 8)];
            if live > 0 {
                assert_eq!(
                    row[live - 1] & !mask,
                    0,
                    "stray bits in last live word at row {r}",
                );
            }
            assert!(
                row[live..].iter().all(|&w| w == 0),
                "padding words nonzero at row {r}",
            );
        }
    }

    #[test]
    fn swap_rows_moves_no_data() {
        let rows = 11;
        let mut m = BitMatrix::from_rows(rows, 100, &row_bits(rows, 100, 0xBEE0)).unwrap();
        let physical_before = m.pitched_buffer().to_vec();
        let map_before: Vec<usize> = (0..rows).map(|r| m.physical_row_index(r)).collect();
        m.swap_rows(4, 9);
        assert_eq!(m.pitched_buffer(), &physical_before[..]);
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
