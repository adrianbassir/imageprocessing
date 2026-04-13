// Core KNN logic: distance computation, neighbor selection, majority vote.
// Pure computation — no I/O, no threading.
// Both sequential and parallel paths call into this module.

use crate::preprocess::NormalizedImage;

/// Euclidean distance between two feature vectors of equal length.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Classify a single query image against the full training set.
/// Returns the predicted label via majority vote among the k nearest neighbors.
pub fn classify(query: &NormalizedImage, train: &[NormalizedImage], k: usize) -> u8 {
    let neighbors = k_nearest(&query.features, train, k);
    majority_vote(&neighbors)
}

/// Select the k nearest training images to `query` by Euclidean distance.
/// Returns (distance, label) pairs sorted ascending by distance.
fn k_nearest(query: &[f32], train: &[NormalizedImage], k: usize) -> Vec<(f32, u8)> {
    let mut distances: Vec<(f32, u8)> = train
        .iter()
        .map(|img| (euclidean_distance(query, &img.features), img.label))
        .collect();

    // Partial sort: only need the k smallest, so use select_nth_unstable for efficiency.
    // Fall back to full sort if k >= len (e.g. in tests with tiny datasets).
    if k < distances.len() {
        distances.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(k);
    }
    distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances
}

/// Majority vote over a slice of (distance, label) pairs.
/// Ties broken by the label of the single nearest neighbor.
fn majority_vote(neighbors: &[(f32, u8)]) -> u8 {
    let mut counts = [0u32; 10]; // CIFAR-10 has 10 classes (0–9)
    for &(_, label) in neighbors {
        counts[label as usize] += 1;
    }

    let max_count = *counts.iter().max().unwrap();
    let winners: Vec<u8> = counts
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c == max_count)
        .map(|(i, _)| i as u8)
        .collect();

    if winners.len() == 1 {
        winners[0]
    } else {
        // Tie: return the label of the single nearest neighbor
        neighbors[0].1
    }
}
