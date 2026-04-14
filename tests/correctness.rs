// Integration tests: validates that the sequential, threaded, and Rayon
// classifiers produce identical predictions on the same inputs.
//
// These tests use a tiny synthetic dataset so they run without CIFAR-10 data.

use std::sync::Arc;

use imgProcessing::parallel::{classify_rayon, classify_threaded};
use imgProcessing::preprocess::NormalizedImage;
use imgProcessing::sequential::classify_sequential;

/// Build a small synthetic dataset of `n` images with `dims` features each.
/// Labels cycle 0..10. Features are random-ish but deterministic.
fn make_dataset(n: usize, dims: usize) -> Vec<NormalizedImage> {
    (0..n)
        .map(|i| NormalizedImage {
            label: (i % 10) as u8,
            features: (0..dims)
                .map(|j| ((i * dims + j) % 256) as f32 / 255.0)
                .collect(),
        })
        .collect()
}

#[test]
fn sequential_and_threaded_agree() {
    let train = make_dataset(200, 32);
    let test = make_dataset(20, 32);
    let k = 3;

    let seq = classify_sequential(&train, &test, k);
    let threaded = classify_threaded(Arc::new(make_dataset(200, 32)), Arc::new(make_dataset(20, 32)), k, 4);

    assert_eq!(seq, threaded, "sequential and threaded-4 predictions differ");
}

#[test]
fn sequential_and_rayon_agree() {
    let train = make_dataset(200, 32);
    let test = make_dataset(20, 32);
    let k = 3;

    let seq = classify_sequential(&train, &test, k);
    let rayon = classify_rayon(&train, &test, k);

    assert_eq!(seq, rayon, "sequential and rayon predictions differ");
}
