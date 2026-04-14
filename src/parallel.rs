// Parallel classification runners.
// Both entry points accept the same inputs and produce the same output type
// as the sequential classifier. Behavioral equivalence is a correctness requirement.

use std::sync::Arc;
use std::thread;

use rayon::prelude::*;

use crate::preprocess::{FlatTrainData, NormalizedImage};
use crate::sequential::classify_sequential;

/// Manual std::thread implementation.
/// Partitions test into num_threads chunks; each thread runs the tiled
/// sequential classifier on its chunk, sharing the training data via Arc.
pub fn classify_threaded(
    train: Arc<FlatTrainData>,
    test: Arc<Vec<NormalizedImage>>,
    k: usize,
    num_threads: usize,
) -> Vec<u8> {
    let num_threads = num_threads.max(1);
    let n = test.len();
    let chunk_size = (n + num_threads - 1) / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let train = Arc::clone(&train);
            let test = Arc::clone(&test);
            let start = t * chunk_size;
            let end = (start + chunk_size).min(n);
            thread::spawn(move || classify_sequential(&train, &test[start..end], k))
        })
        .collect();

    handles
        .into_iter()
        .flat_map(|h| h.join().expect("thread panicked"))
        .collect()
}

/// Rayon data-parallel implementation.
/// Splits the test slice into chunks, writes results directly into a
/// pre-allocated output buffer to avoid intermediate Vec allocations.
pub fn classify_rayon(train: &FlatTrainData, test: &[NormalizedImage], k: usize) -> Vec<u8> {
    const RAYON_CHUNK: usize = 16;
    let mut preds = vec![0u8; test.len()];
    preds.par_chunks_mut(RAYON_CHUNK)
        .zip(test.par_chunks(RAYON_CHUNK))
        .for_each(|(out, chunk)| {
            let results = classify_sequential(train, chunk, k);
            out.copy_from_slice(&results);
        });
    preds
}
