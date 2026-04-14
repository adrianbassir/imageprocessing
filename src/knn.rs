// Core KNN logic: distance computation, neighbor selection, majority vote.
// Pure computation — no I/O, no threading.
// Both sequential and parallel paths call into this module.

use crate::preprocess::NormalizedImage;

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
/// If the running sum exceeds `threshold`, returns `threshold` immediately.
/// Used internally by k_nearest to skip candidates that can't be in the top-k.
#[inline]
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
pub fn classify(query: &NormalizedImage, train: &[NormalizedImage], k: usize) -> u8 {
    let neighbors = k_nearest(&query.features, train, k);
    majority_vote(&neighbors)
}

/// Select the k nearest training images to `query` by squared Euclidean distance.
/// Returns (distance, label) pairs sorted ascending by distance.
/// Uses a fixed-size max-heap to avoid allocating all distances up front,
/// and early abandonment to skip candidates outside the current top-k.
fn k_nearest(query: &[f32], train: &[NormalizedImage], k: usize) -> Vec<(f32, u8)> {
    // Use a fixed-capacity buffer as a max-heap (largest distance at top).
    // Once full, any candidate with distance >= heap max is skipped entirely.
    let mut heap: Vec<(f32, u8)> = Vec::with_capacity(k + 1);

    for img in train {
        // Threshold: if heap is full, use worst distance so far; else infinity.
        let threshold = if heap.len() == k {
            heap[0].0 // max-heap: root is the largest distance
        } else {
            f32::INFINITY
        };

        let dist = squared_distance_bounded(query, &img.features, threshold);

        if dist < threshold {
            // Push and re-heapify as a max-heap by distance
            heap.push((dist, img.label));
            // Bubble up to maintain max-heap invariant
            let mut i = heap.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if heap[parent].0 < heap[i].0 {
                    heap.swap(parent, i);
                    i = parent;
                } else {
                    break;
                }
            }
            // If over capacity, remove the root (largest)
            if heap.len() > k {
                let last = heap.len() - 1;
                heap.swap(0, last);
                heap.pop();
                // Sift down to restore max-heap
                let mut i = 0;
                loop {
                    let left = 2 * i + 1;
                    let right = 2 * i + 2;
                    let mut largest = i;
                    if left < heap.len() && heap[left].0 > heap[largest].0 {
                        largest = left;
                    }
                    if right < heap.len() && heap[right].0 > heap[largest].0 {
                        largest = right;
                    }
                    if largest == i {
                        break;
                    }
                    heap.swap(i, largest);
                    i = largest;
                }
            }
        }
    }

    // Sort ascending by distance for majority_vote
    heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap
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
