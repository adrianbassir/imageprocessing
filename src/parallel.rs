// Parallel classification runners.
// Both entry points accept the same inputs and produce the same output type
// as the sequential classifier. Behavioral equivalence is a correctness requirement.

use std::sync::Arc;
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
    todo!()
}

/// Rayon data-parallel implementation.
/// Uses par_iter() over the test slice; each classification is an independent
/// work unit scheduled by Rayon's work-stealing runtime.
pub fn classify_rayon(train: &[NormalizedImage], test: &[NormalizedImage], k: usize) -> Vec<u8> {
    todo!()
}
