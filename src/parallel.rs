// Parallel classification runners.
// Both entry points accept the same inputs and produce the same output type
// as the sequential classifier. Behavioral equivalence is a correctness requirement.
//
// Note: requires `rayon` in Cargo.toml for classify_rayon,
// and `std::thread` (stdlib) for classify_threaded.

use std::sync::Arc;
use std::thread;

use rayon::prelude::*;

use crate::knn::classify;
use crate::preprocess::NormalizedImage;

/// Manual std::thread implementation.
/// Partitions `test` into `num_threads` equal chunks; each thread holds an
/// Arc reference to the full training set. Results are collected after all
/// threads join — the only synchronization point.
pub fn classify_threaded(
    train: Arc<Vec<NormalizedImage>>,
    test: &[NormalizedImage],
    k: usize,
    num_threads: usize,
) -> Vec<u8> {
    let num_threads = num_threads.max(1);
    let chunk_size = (test.len() + num_threads - 1) / num_threads;

    // We need to send slices of `test` to threads. Since NormalizedImage doesn't
    // implement Copy, we clone each chunk into its own Vec so each thread owns its data.
    let chunks: Vec<Vec<NormalizedImage>> = test
        .chunks(chunk_size)
        .map(|chunk| {
            chunk
                .iter()
                .map(|img| NormalizedImage {
                    label: img.label,
                    features: img.features.clone(),
                })
                .collect()
        })
        .collect();

    let handles: Vec<_> = chunks
        .into_iter()
        .map(|chunk| {
            let train = Arc::clone(&train);
            thread::spawn(move || {
                chunk
                    .iter()
                    .map(|img| classify(img, &train, k))
                    .collect::<Vec<u8>>()
            })
        })
        .collect();

    handles
        .into_iter()
        .flat_map(|h| h.join().expect("thread panicked"))
        .collect()
}

/// Rayon data-parallel implementation.
/// Uses par_iter() over the test slice; each classification is an independent
/// work unit scheduled by Rayon's work-stealing runtime.
pub fn classify_rayon(train: &[NormalizedImage], test: &[NormalizedImage], k: usize) -> Vec<u8> {
    test.par_iter()
        .map(|img| classify(img, train, k))
        .collect()
}
