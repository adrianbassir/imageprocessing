// Criterion benchmark harness for KNN classification.
// Run with: cargo bench
//
// Uses a synthetic dataset (no CIFAR-10 files required) so benchmarks
// work out of the box. Dimensions and dataset sizes mirror realistic
// CIFAR-10 proportions at a reduced scale to keep bench time manageable.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use imgProcessing::parallel::{classify_rayon, classify_threaded};
use imgProcessing::preprocess::NormalizedImage;
use imgProcessing::sequential::classify_sequential;

const DIMS: usize = 3072; // CIFAR-10 feature vector length
const TRAIN_SIZE: usize = 2000;
const TEST_SIZE: usize = 100;
const K: usize = 5;

/// Build a deterministic synthetic dataset. Features are pseudo-random but
/// reproducible so benchmark results are comparable across runs.
fn make_dataset(n: usize, dims: usize, seed_offset: usize) -> Vec<NormalizedImage> {
    (0..n)
        .map(|i| NormalizedImage {
            label: (i % 10) as u8,
            features: (0..dims)
                .map(|j| {
                    // Simple deterministic hash → float in [0, 1]
                    let v = (i.wrapping_mul(1_000_003).wrapping_add(j).wrapping_add(seed_offset))
                        % 256;
                    v as f32 / 255.0
                })
                .collect(),
        })
        .collect()
}

fn bench_sequential(c: &mut Criterion) {
    let train = make_dataset(TRAIN_SIZE, DIMS, 0);
    let test = make_dataset(TEST_SIZE, DIMS, 1);

    c.bench_function("sequential", |b| {
        b.iter(|| classify_sequential(&train, &test, K))
    });
}

fn bench_threaded(c: &mut Criterion) {
    let train = Arc::new(make_dataset(TRAIN_SIZE, DIMS, 0));
    let test = make_dataset(TEST_SIZE, DIMS, 1);

    let mut group = c.benchmark_group("threaded");

    for num_threads in [1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &n| {
                b.iter(|| classify_threaded(Arc::clone(&train), &test, K, n))
            },
        );
    }

    group.finish();
}

fn bench_rayon(c: &mut Criterion) {
    let train = make_dataset(TRAIN_SIZE, DIMS, 0);
    let test = make_dataset(TEST_SIZE, DIMS, 1);

    c.bench_function("rayon", |b| {
        b.iter(|| classify_rayon(&train, &test, K))
    });
}

criterion_group!(benches, bench_sequential, bench_threaded, bench_rayon);
criterion_main!(benches);
