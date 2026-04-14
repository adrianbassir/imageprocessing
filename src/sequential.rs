// Single-threaded classification runner.
// Reference implementation — all parallel results are validated against this.

use crate::knn::{heap_push, heap_max, knn_vote};
use crate::preprocess::{FlatTrainData, NormalizedImage};

/// Classify every image in `test` against `train` using k nearest neighbors.
/// Returns predicted labels in the same order as `test`.
///
/// Uses tiled execution: processes TILE test images per pass over training data.
/// Each training image's features are loaded once and used for TILE distance
/// computations, amortizing memory bandwidth across multiple queries.
pub fn classify_sequential(train: &FlatTrainData, test: &[NormalizedImage], k: usize) -> Vec<u8> {
    const TILE: usize = 4;
    let mut predictions = vec![0u8; test.len()];

    for tile_start in (0..test.len()).step_by(TILE) {
        let tile_end = (tile_start + TILE).min(test.len());
        let tile = &test[tile_start..tile_end];
        let tile_size = tile.len();

        // Per-tile heaps and thresholds — one per test image in the tile
        let mut heaps: [Vec<(f32, u8)>; TILE] = Default::default();
        let mut thresholds = [f32::INFINITY; TILE];
        for h in heaps.iter_mut() {
            h.reserve(k + 1);
        }

        // Single pass over all training images, updating all TILE heaps
        for ti in 0..train.n {
            let train_feat = train.features_of(ti);
            let label = train.labels[ti];

            for b in 0..tile_size {
                let dist = crate::knn::squared_distance_bounded(
                    &tile[b].features,
                    train_feat,
                    thresholds[b],
                );
                if dist < thresholds[b] {
                    heap_push(&mut heaps[b], dist, label, k);
                    thresholds[b] = heap_max(&heaps[b]);
                }
            }
        }

        for b in 0..tile_size {
            heaps[b].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            predictions[tile_start + b] = knn_vote(&heaps[b]);
        }
    }

    predictions
}
