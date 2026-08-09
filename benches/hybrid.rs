//! Sparse hybrid solve versus a full dense decomposition on an RFC 6330
//! LT-degree-shaped system.

use core::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use fgf::Gf8;
use fgf::field::Field;
use gfm::{Hybrid, Matrix, Ple, PleScratch, SolveScratch};

const DEGREE_THRESHOLDS: [u32; 31] = [
    0, 5_243, 529_531, 704_294, 791_675, 844_104, 879_057, 904_023, 922_747, 937_311, 948_962,
    958_494, 966_438, 973_160, 978_921, 983_914, 988_283, 992_138, 995_565, 998_631, 1_001_391,
    1_003_887, 1_006_157, 1_008_229, 1_010_129, 1_011_876, 1_013_490, 1_014_983, 1_016_370,
    1_017_662, 1_048_576,
];

fn draw(state: &mut u64, modulo: usize) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    if modulo == 0 {
        0
    } else {
        (*state >> 33) as usize % modulo
    }
}

fn degree(value: usize, columns: usize) -> usize {
    DEGREE_THRESHOLDS
        .partition_point(|&threshold| threshold <= value as u32)
        .min(columns.saturating_sub(2).max(1))
}

fn support(columns: usize, weight: usize, state: &mut u64) -> Vec<u32> {
    let mut result = Vec::with_capacity(weight);
    while result.len() < weight {
        let column = draw(state, columns) as u32;
        if !result.contains(&column) {
            result.push(column);
        }
    }
    result.sort_unstable();
    result
}

fn system(columns: usize, symbol_bytes: usize) -> (Hybrid<Gf8>, Matrix<Gf8>, Matrix<Gf8>) {
    let mut state = 0x6330_1000;
    let overhead = (columns as f64).sqrt().ceil() as usize + 8;
    let rows = columns + overhead;
    let mut hybrid = Hybrid::<Gf8>::new(columns, symbol_bytes);
    let mut dense = Matrix::<Gf8>::zeros(rows, columns).unwrap();
    let rhs = Matrix::<Gf8>::zeros(rows, symbol_bytes).unwrap();
    let zero_rhs = vec![0; symbol_bytes];
    for row in 0..rows {
        let row_support = support(
            columns,
            degree(draw(&mut state, 1 << 20), columns),
            &mut state,
        );
        hybrid.push_binary_row(&row_support, &zero_rhs);
        for &column in &row_support {
            dense.set(row, column as usize, <Gf8 as Field>::Elem::ONE);
        }
    }
    (hybrid, dense, rhs)
}

fn benchmark(c: &mut Criterion) {
    const K: usize = 1_000;
    const SYMBOL_BYTES: usize = 1_024;
    let (mut hybrid, dense, rhs) = system(K, SYMBOL_BYTES);
    let mut hybrid_values = Matrix::<Gf8>::zeros(K, SYMBOL_BYTES).unwrap();
    let mut determined = vec![false; K];
    hybrid
        .solve_into(&mut hybrid_values, &mut determined)
        .unwrap();

    let mut ple_scratch = PleScratch::new();
    let mut solve_scratch = SolveScratch::new();
    let mut dense_values = Matrix::<Gf8>::zeros(K, SYMBOL_BYTES).unwrap();

    let mut group = c.benchmark_group("raptorq_shaped_k1000");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("hybrid", |b| {
        b.iter(|| {
            black_box(
                hybrid
                    .solve_into(&mut hybrid_values, &mut determined)
                    .unwrap(),
            )
        });
    });
    group.bench_function("dense_ple", |b| {
        b.iter(|| {
            let ple = Ple::decompose(black_box(dense.clone()), &mut ple_scratch);
            ple.solve_into(&rhs, &mut dense_values, &mut solve_scratch)
                .unwrap();
            black_box(&dense_values);
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
