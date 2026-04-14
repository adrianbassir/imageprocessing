// Core KNN logic: distance computation, neighbor selection, majority vote.
// Pure computation — no I/O, no threading.
// Both sequential and parallel paths call into this module.

use crate::preprocess::{FlatTrainData, NormalizedImage};

/// Squared Euclidean distance between two feature vectors.
/// Skipping sqrt is valid for ranking — sqrt is monotonic so order is preserved.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Squared Euclidean distance with early abandonment.
/// Bails out as soon as the partial sum exceeds `threshold`.
#[inline(always)]
fn squared_distance_bounded(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y) * (x - y);
        if sum >= threshold {
            return threshold;
        }
    }
    sum
}

/// Classify a single query image against the full training set.
/// Returns the predicted label via majority vote among the k nearest neighbors.
pub fn classify(query: &NormalizedImage, train: &FlatTrainData, k: usize) -> u8 {
    let neighbors = k_nearest(&query.features, train, k);
    majority_vote(&neighbors)
}

/// Select the k nearest training images to `query` by squared Euclidean distance.
/// Uses a fixed-size max-heap with early abandonment to skip distant candidates.
fn k_nearest(query: &[f32], train: &FlatTrainData, k: usize) -> Vec<(f32, u8)> {
    let mut heap: Vec<(f32, u8)> = Vec::with_capacity(k + 1);

    for i in 0..train.n {
        let threshold = if heap.len() == k { heap[0].0 } else { f32::INFINITY };
        let dist = squared_distance_bounded(query, train.features_of(i), threshold);

        if dist < threshold {
            heap.push((dist, train.labels[i]));
            // Bubble up
            let mut idx = heap.len() - 1;
            while idx > 0 {
                let parent = (idx - 1) / 2;
                if heap[parent].0 < heap[idx].0 {
                    heap.swap(parent, idx);
                    idx = parent;
                } else {
                    break;
                }
            }
            // Remove root (largest) if over capacity
            if heap.len() > k {
                let last = heap.len() - 1;
                heap.swap(0, last);
                heap.pop();
                // Sift down
                let mut idx = 0;
                loop {
                    let left = 2 * idx + 1;
                    let right = 2 * idx + 2;
                    let mut largest = idx;
                    if left < heap.len() && heap[left].0 > heap[largest].0 { largest = left; }
                    if right < heap.len() && heap[right].0 > heap[largest].0 { largest = right; }
                    if largest == idx { break; }
                    heap.swap(idx, largest);
                    idx = largest;
                }
            }
        }
    }

    heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap
}

/// Majority vote over a slice of (distance, label) pairs.
/// Ties broken by the label of the single nearest neighbor.
fn majority_vote(neighbors: &[(f32, u8)]) -> u8 {
    let mut counts = [0u32; 10];
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

    if winners.len() == 1 { winners[0] } else { neighbors[0].1 }
}
