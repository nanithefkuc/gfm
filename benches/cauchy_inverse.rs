//! Closed-form Cauchy inverse versus the general `Ple` inverse.
//!
//! The O(k²) claim is only worth making if the closed form actually beats the
//! O(k³) elimination across the useful range. Both paths compute a full
//! `k × k` inverse from scratch; the numbers land in `BENCHMARKS.md`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fgf::Gf8;
use gfm::{Cauchy, Matrix, Ple, PleScratch};
use std::hint::black_box;

fn cauchy_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cauchy_inverse");
    for k in [4usize, 8, 16, 32, 48, 64] {
        let cauchy = Cauchy::<Gf8>::indexed(k, k).expect("2k <= 256");
        let mut mat = Matrix::<Gf8>::zeros(k, k).unwrap();
        cauchy.materialize_into(&mut mat);
        let mut out = Matrix::<Gf8>::zeros(k, k).unwrap();

        group.bench_with_input(BenchmarkId::new("closed_form", k), &k, |b, _| {
            b.iter(|| {
                cauchy.inverse_into(&mut out);
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("ple", k), &k, |b, _| {
            b.iter(|| {
                let ple = Ple::decompose(mat.clone(), &mut PleScratch::new());
                ple.inverse_into(&mut out).unwrap();
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, cauchy_inverse);
criterion_main!(benches);
