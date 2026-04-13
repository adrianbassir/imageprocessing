// Criterion benchmark harness for KNN classification.
// Run with: cargo bench

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_sequential(c: &mut Criterion) {
    todo!()
}

fn bench_threaded(c: &mut Criterion) {
    todo!()
}

fn bench_rayon(c: &mut Criterion) {
    todo!()
}

criterion_group!(benches, bench_sequential, bench_threaded, bench_rayon);
criterion_main!(benches);
