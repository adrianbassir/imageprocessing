// Parallel classification runners.
// Both entry points accept the same inputs and produce the same output type
// as the sequential classifier. Behavioral equivalence is a correctness requirement.

use std::sync::Arc;
use std::thread;

use rayon::prelude::*;

use crate::knn::classify;
use crate::preprocess::NormalizedImage;

/// Manual std::thread implementation.
/// Both train and test are wrapped in Arc to avoid cloning data across threads.
/// Each thread receives index bounds into the shared test slice instead of owned chunks.
/// Results are pre-allocated and written by index, then collected after all threads join.
pub fn classify_threaded(
    train: Arc<Vec<NormalizedImage>>,
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
            thread::spawn(move || {
                test[start..end]
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
