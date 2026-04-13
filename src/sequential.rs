// Single-threaded classification runner.
// Reference implementation — all parallel results are validated against this.

use crate::preprocess::NormalizedImage;

/// Classify every image in `test` against `train` using k nearest neighbors.
/// Returns predicted labels in the same order as `test`.
pub fn classify_sequential(train: &[NormalizedImage], test: &[NormalizedImage], k: usize) -> Vec<u8> {
    todo!()
}
