#![allow(clippy::needless_range_loop)]

//! A/B twins for elimination dispatch, M4RI slabs, and the small-matrix path.

use core::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fgf::field::Field;
use fgf::{FieldKernels, Gf8, Gf16};
use gfm::bits::{Ple as BitPle, PleScratch as BitPleScratch};
use gfm::{BitMatrix, Matrix, Ple, PleScratch, SmallMatrix, SolveScratch};

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    *state
}

fn dense_matrix<F: FieldKernels>(n: usize, seed: u64) -> Matrix<F> {
    dense_rect::<F>(n, n, seed)
}

fn dense_rect<F: FieldKernels>(rows: usize, cols: usize, seed: u64) -> Matrix<F> {
    let mut state = seed;
    let mut matrix = Matrix::<F>::zeros(rows, cols).unwrap();
    for row in 0..rows {
        for col in 0..cols {
            let bytes = next(&mut state).to_le_bytes();
            matrix.set(row, col, F::read(&bytes[..F::BYTES]));
        }
    }
    matrix
}

fn bit_matrix(n: usize, seed: u64) -> BitMatrix {
    let mut state = seed;
    let mut matrix = BitMatrix::zeros(n, n).unwrap();
    for row in 0..n {
        for col in 0..n {
            matrix.set(row, col, next(&mut state) >> 63 != 0);
        }
    }
    matrix
}

fn bench_dense<F: FieldKernels>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("dense_dispatch_{}", F::NAME));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for n in [32usize, 64, 128, 256] {
        let matrix = dense_matrix::<F>(n, 0xD15A ^ n as u64);
        group.bench_with_input(BenchmarkId::new("axpy", n), &n, |b, _| {
            let mut scratch = PleScratch::new();
            b.iter(|| {
                black_box(Ple::decompose_with_panel_width(
                    black_box(matrix.clone()),
                    &mut scratch,
                    1,
                ));
            });
        });
        for (name, width) in [("blocked_8", 8), ("blocked_16", 16), ("blocked_64", 64)] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
                let mut scratch = PleScratch::new();
                b.iter(|| {
                    black_box(Ple::decompose_with_panel_width(
                        black_box(matrix.clone()),
                        &mut scratch,
                        width,
                    ));
                });
            });
        }
    }
    let control = dense_matrix::<F>(128, 0xC017);
    for name in ["control_a", "control_b"] {
        group.bench_function(name, |b| {
            let mut scratch = PleScratch::new();
            b.iter(|| {
                black_box(Ple::decompose_with_panel_width(
                    black_box(control.clone()),
                    &mut scratch,
                    1,
                ));
            });
        });
    }
    group.finish();
}

fn bench_bits(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_m4ri");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for n in [64usize, 128, 256, 512, 1024] {
        let matrix = bit_matrix(n, 0x4A41 ^ n as u64);
        group.bench_with_input(BenchmarkId::new("plain", n), &n, |b, _| {
            let mut scratch = BitPleScratch::new();
            b.iter(|| {
                black_box(BitPle::decompose_plain(
                    black_box(matrix.clone()),
                    &mut scratch,
                ));
            });
        });
        group.bench_with_input(BenchmarkId::new("m4ri", n), &n, |b, _| {
            let mut scratch = BitPleScratch::new();
            b.iter(|| {
                black_box(BitPle::decompose_m4ri(
                    black_box(matrix.clone()),
                    &mut scratch,
                ));
            });
        });
    }
    group.finish();
}
fn bench_newton_john(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_newton_john");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for n in [128usize, 256, 512, 1024] {
        let matrix = dense_matrix::<Gf8>(n, 0x4E4A ^ n as u64);
        group.bench_with_input(BenchmarkId::new("blocked", n), &n, |b, _| {
            let mut scratch = PleScratch::new();
            b.iter(|| {
                black_box(Ple::decompose(black_box(matrix.clone()), &mut scratch));
            });
        });
        group.bench_with_input(BenchmarkId::new("newton_john", n), &n, |b, _| {
            let mut scratch = PleScratch::new();
            b.iter(|| {
                black_box(Ple::decompose_newton_john(
                    black_box(matrix.clone()),
                    &mut scratch,
                ));
            });
        });
    }
    group.finish();
}

fn full_rank_gf8<const K: usize>() -> Matrix<Gf8> {
    let mut state = 0x5A11 ^ K as u64;
    let mut lower = [[<Gf8 as Field>::Elem::ZERO; K]; K];
    let mut upper = lower;
    for row in 0..K {
        lower[row][row] = <Gf8 as Field>::Elem::ONE;
        upper[row][row] = <Gf8 as Field>::Elem::ONE;
        for col in 0..row {
            lower[row][col] = Gf8::read(&[(next(&mut state) >> 56) as u8]);
        }
        for col in (row + 1)..K {
            upper[row][col] = Gf8::read(&[(next(&mut state) >> 56) as u8]);
        }
    }
    let mut matrix = Matrix::<Gf8>::zeros(K, K).unwrap();
    for row in 0..K {
        for col in 0..K {
            let mut value = <Gf8 as Field>::Elem::ZERO;
            for term in 0..K {
                value = value.add(lower[row][term].mul(upper[term][col]));
            }
            matrix.set(row, col, value);
        }
    }
    matrix
}

fn bench_small_order<const K: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let matrix = full_rank_gf8::<K>();
    let rhs = dense_rect::<Gf8>(K, 1024, 0x0052_4853 ^ K as u64);
    let mut out = Matrix::<Gf8>::zeros(K, 1024).unwrap();
    assert_eq!(
        SmallMatrix::<Gf8, K>::from_matrix(&matrix).rank(),
        K,
        "benchmark matrix must be full rank"
    );
    group.bench_with_input(BenchmarkId::new("small", K), &K, |b, _| {
        b.iter(|| {
            let small = SmallMatrix::<Gf8, K>::from_matrix(black_box(&matrix));
            small
                .solve_into(black_box(&rhs), black_box(&mut out))
                .unwrap();
            black_box(&out);
        });
    });
    group.bench_with_input(BenchmarkId::new("ple", K), &K, |b, _| {
        let mut ple_scratch = PleScratch::new();
        let mut solve_scratch = SolveScratch::new();
        b.iter(|| {
            let ple = Ple::decompose(black_box(matrix.clone()), &mut ple_scratch);
            ple.solve_into(black_box(&rhs), black_box(&mut out), &mut solve_scratch)
                .unwrap();
            black_box(&out);
        });
    });
}

macro_rules! bench_small_orders {
    ($group:expr; $($k:literal),+ $(,)?) => {
        $(bench_small_order::<$k>($group);)+
    };
}

fn bench_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_matrix");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    bench_small_orders!(
        &mut group;
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    );
    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    bench_dense::<Gf8>(c);
    bench_dense::<Gf16>(c);
    bench_newton_john(c);
    bench_bits(c);
    bench_small(c);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
