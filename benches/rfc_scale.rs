//! Scheduler cost on RFC 6330-shaped systems: LT-degree rows alone, and
//! LT rows plus a dense HDPC-style field band (the shape that makes the
//! generic schedule quadratic). Slow configurations run through
//! `iter_custom` single-shot timing instead of criterion's sampling.

use core::hint::black_box;
use std::cell::RefCell;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fgf::Gf8;
use fgf::field::Field;
use gfm::Hybrid;

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

/// An RFC 6330-shaped system: `columns` LT-degree binary rows plus
/// overhead, optionally a dense `band`-row HDPC-style field block over
/// every column (coefficient powers of alpha), with the trailing `band`
/// columns pre-inactivated as RaptorQ PI columns.
fn system(columns: usize, band: usize, symbol_bytes: usize, defer_band: bool) -> Hybrid<Gf8> {
    let mut state = 0x6330_5CA1u64;
    let overhead = (columns as f64).sqrt().ceil() as usize + 8;
    let initial: Vec<u32> = ((columns - band)..columns).map(|c| c as u32).collect();
    let mut hybrid = Hybrid::<Gf8>::with_initial_inactive(columns, symbol_bytes, &initial);
    let zero_rhs = vec![0u8; symbol_bytes];
    for _ in 0..(columns + overhead) {
        let row_support = support(
            columns - band,
            degree(draw(&mut state, 1 << 20), columns - band),
            &mut state,
        );
        hybrid.push_binary_row(&row_support, &zero_rhs);
    }
    if band > 0 {
        let alpha = <Gf8 as Field>::read(&[2]);
        let mut coefficient = <Gf8 as Field>::Elem::ONE;
        let coeffs: Vec<_> = (0..columns)
            .map(|_| {
                let c = coefficient;
                coefficient = coefficient.mul(alpha);
                c
            })
            .collect();
        let support: Vec<u32> = (0..columns as u32).collect();
        for _ in 0..band {
            if defer_band {
                hybrid.push_deferred_field_row(&support, &coeffs, &zero_rhs);
            } else {
                hybrid.push_field_row(&support, &coeffs, &zero_rhs);
            }
        }
    }
    hybrid
}

fn solve(hybrid: &mut Hybrid<Gf8>, columns: usize, symbol_bytes: usize) {
    let mut values = gfm::Matrix::<Gf8>::zeros(columns, symbol_bytes).unwrap();
    let mut determined = vec![false; columns];
    let rank = hybrid.solve_into(&mut values, &mut determined).unwrap();
    black_box((rank, &values, &determined));
}

fn rfc_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfc_scale");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(1));
    for &columns in &[1_000usize, 5_000, 10_000, 20_000] {
        let hybrid = RefCell::new(system(columns, 0, 64, false));
        group.bench_with_input(
            BenchmarkId::new("lt_only", columns),
            &hybrid,
            |b, hybrid| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        solve(&mut hybrid.borrow_mut(), columns, 64);
                    }
                    start.elapsed()
                })
            },
        );
    }
    for &columns in &[500usize, 1_000, 2_000, 4_000] {
        let band = 8 + columns / 20;
        for (name, defer_band) in [("lt_hdpc_eager", false), ("lt_hdpc_deferred", true)] {
            let hybrid = RefCell::new(system(columns, band, 64, defer_band));
            group.bench_with_input(BenchmarkId::new(name, columns), &hybrid, |b, hybrid| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        solve(&mut hybrid.borrow_mut(), columns, 64);
                    }
                    start.elapsed()
                })
            });
        }
    }
    // The deferred shape stays tractable at the RFC 6330 maximum.
    {
        let columns = 56_403usize;
        let band = 108;
        let hybrid = RefCell::new(system(columns, band, 64, true));
        group.bench_with_input(
            BenchmarkId::new("lt_hdpc_deferred", columns),
            &hybrid,
            |b, hybrid| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        solve(&mut hybrid.borrow_mut(), columns, 64);
                    }
                    start.elapsed()
                })
            },
        );
    }
    for &(columns, defer_band) in &[
        (500usize, true),
        (1_000, true),
        (2_000, true),
        (4_000, true),
        (20_000, true),
        (56_403, true),
    ] {
        let band = 8 + columns / 20;
        let hybrid = RefCell::new(system(columns, band, 64, defer_band));
        group.bench_with_input(
            BenchmarkId::new(
                if defer_band {
                    "lt_hdpc_deferred"
                } else {
                    "lt_hdpc_eager"
                },
                columns,
            ),
            &hybrid,
            |b, hybrid| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        solve(&mut hybrid.borrow_mut(), columns, 64);
                    }
                    start.elapsed()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, rfc_scale);
criterion_main!(benches);
