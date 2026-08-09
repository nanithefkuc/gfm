//! Symbol-axis row-update scaling through fixed-size Rayon pools.

use core::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fgf::Gf8;
use fgf::field::Field;
use gfm::benchmark_mul_add;
use rayon::ThreadPoolBuilder;

const ROW_BYTES: usize = 8 * 1024 * 1024;

fn benchmark(c: &mut Criterion) {
    let src = vec![0xA7; ROW_BYTES];
    let mut dst = vec![0x39; ROW_BYTES];
    let factor = Gf8::read(&[0x53]);
    let mut group = c.benchmark_group("parallel_row_gf8_8mib");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for threads in [1usize, 2, 4, 8, 16] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, _| {
            b.iter(|| {
                pool.install(|| {
                    benchmark_mul_add::<Gf8>(
                        black_box(&mut dst),
                        black_box(factor),
                        black_box(&src),
                    );
                });
                black_box(&dst);
            });
        });
    }
    group.finish();
}

fn threshold(c: &mut Criterion) {
    let serial = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let parallel = ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let factor = Gf8::read(&[0x53]);
    let mut group = c.benchmark_group("parallel_row_threshold_gf8");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for bytes in [64, 128, 256, 512, 1024, 2048, 4096, 8192].map(|kib| kib * 1024) {
        let src = vec![0xA7; bytes];
        let mut dst = vec![0x39; bytes];
        for (name, pool) in [("serial", &serial), ("parallel_8", &parallel)] {
            group.bench_with_input(BenchmarkId::new(name, bytes), &bytes, |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        benchmark_mul_add::<Gf8>(
                            black_box(&mut dst),
                            black_box(factor),
                            black_box(&src),
                        );
                    });
                    black_box(&dst);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark, threshold);
criterion_main!(benches);
