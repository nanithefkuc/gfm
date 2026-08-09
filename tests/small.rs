//! Compact small-matrix equivalence against the general dense decomposition.

mod common;

use common::{draw, noise};
use fgf::field::Field;
use fgf::{Gf8, Gf16};
use gfm::{Matrix, Ple, PleScratch, SmallMatrix, SolveScratch};

fn matrix<const K: usize>(seed: u64, deficient: bool) -> Matrix<Gf8> {
    let mut state = seed | 1;
    let mut matrix = Matrix::<Gf8>::zeros(K, K).unwrap();
    for row in 0..K {
        for col in 0..K {
            let value = if deficient && (row + 1 == K || col + 1 == K) {
                <Gf8 as Field>::Elem::ZERO
            } else {
                Gf8::read(&[draw(&mut state, 256) as u8])
            };
            matrix.set(row, col, value);
        }
    }
    matrix
}

fn check_rank<const K: usize>() {
    for deficient in [false, true] {
        let matrix = matrix::<K>(0x5A11 ^ K as u64, deficient);
        let small = SmallMatrix::<Gf8, K>::from_matrix(&matrix);
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
                Gf16::read(&(draw(&mut state, u16::MAX as usize) as u16).to_le_bytes()),
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
fn rank_agrees_with_ple_at_every_order() {
    every_order!(check_rank);
}

#[test]
fn solve_agrees_with_ple_at_every_order() {
    every_order!(check_solve);
}
