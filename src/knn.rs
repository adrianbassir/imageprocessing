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

/// Squared Euclidean distance with chunked early abandonment.
///
/// Processes 16 floats per chunk with no branch inside — the compiler can emit
/// AVX2 instructions across the full chunk. Threshold is checked once per chunk
/// (every 16 floats) rather than every float, giving 192 checks instead of 3072
/// while still terminating early on distant candidates.
///
/// 8 independent accumulators break the reduction dependency chain so the CPU
/// can execute multiple FMA operations in parallel within each chunk.
#[inline(always)]
pub fn squared_distance_bounded(a: &[f32], b: &[f32], threshold: f32) -> f32 {
    const CHUNK: usize = 16;
    let full_chunks = a.len() / CHUNK;
    let mut sum = 0.0f32;

    for c in 0..full_chunks {
        let base = c * CHUNK;
        let a_chunk = &a[base..base + CHUNK];
        let b_chunk = &b[base..base + CHUNK];

        // 8 independent accumulators — breaks the FP reduction dependency chain.
        // LLVM can schedule these as independent FMA lanes.
        let mut acc0 = 0.0f32; let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32; let mut acc3 = 0.0f32;
        let mut acc4 = 0.0f32; let mut acc5 = 0.0f32;
        let mut acc6 = 0.0f32; let mut acc7 = 0.0f32;

        let d0 = a_chunk[0]  - b_chunk[0];  acc0 += d0 * d0;
        let d1 = a_chunk[1]  - b_chunk[1];  acc1 += d1 * d1;
        let d2 = a_chunk[2]  - b_chunk[2];  acc2 += d2 * d2;
        let d3 = a_chunk[3]  - b_chunk[3];  acc3 += d3 * d3;
        let d4 = a_chunk[4]  - b_chunk[4];  acc4 += d4 * d4;
        let d5 = a_chunk[5]  - b_chunk[5];  acc5 += d5 * d5;
        let d6 = a_chunk[6]  - b_chunk[6];  acc6 += d6 * d6;
        let d7 = a_chunk[7]  - b_chunk[7];  acc7 += d7 * d7;
        let d8 = a_chunk[8]  - b_chunk[8];  acc0 += d8 * d8;
        let d9 = a_chunk[9]  - b_chunk[9];  acc1 += d9 * d9;
        let d10 = a_chunk[10] - b_chunk[10]; acc2 += d10 * d10;
        let d11 = a_chunk[11] - b_chunk[11]; acc3 += d11 * d11;
        let d12 = a_chunk[12] - b_chunk[12]; acc4 += d12 * d12;
        let d13 = a_chunk[13] - b_chunk[13]; acc5 += d13 * d13;
        let d14 = a_chunk[14] - b_chunk[14]; acc6 += d14 * d14;
        let d15 = a_chunk[15] - b_chunk[15]; acc7 += d15 * d15;

        sum += (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

        if sum >= threshold {
            return threshold;
        }
    }

    // Remainder (3072 % 16 == 0 for CIFAR-10, so this path is rarely taken)
    for i in (full_chunks * CHUNK)..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
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

/// Push a (distance, label) pair into a max-heap of capacity k.
/// If the heap exceeds k, the largest element is removed.
pub fn heap_push(heap: &mut Vec<(f32, u8)>, dist: f32, label: u8, k: usize) {
    heap.push((dist, label));
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
    if heap.len() > k {
        let last = heap.len() - 1;
        heap.swap(0, last);
        heap.pop();
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

/// Return the current worst (largest) distance in the heap, or infinity if empty.
#[inline(always)]
pub fn heap_max(heap: &[(f32, u8)]) -> f32 {
    if heap.is_empty() { f32::INFINITY } else { heap[0].0 }
}

/// Majority vote over a sorted (distance, label) slice — tie-break to nearest.
pub fn knn_vote(neighbors: &[(f32, u8)]) -> u8 {
    majority_vote(neighbors)
}

/// Select the k nearest training images to `query` by squared Euclidean distance.
/// Uses a fixed-size max-heap with early abandonment to skip distant candidates.
fn k_nearest(query: &[f32], train: &FlatTrainData, k: usize) -> Vec<(f32, u8)> {
    let mut heap: Vec<(f32, u8)> = Vec::with_capacity(k + 1);

    for i in 0..train.n {
        let threshold = heap_max(&heap).min(if heap.len() == k { heap[0].0 } else { f32::INFINITY });
        let dist = squared_distance_bounded(query, train.features_of(i), threshold);
        if dist < threshold {
            heap_push(&mut heap, dist, train.labels[i], k);
        }
    }

    heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap
}

/// Majority vote over a slice of (distance, label) pairs.
/// Ties broken by the label of the single nearest neighbor.
/// Uses a fixed [u32; 10] array — no heap allocation per call.
fn majority_vote(neighbors: &[(f32, u8)]) -> u8 {
    let mut counts = [0u32; 10];
    for &(_, label) in neighbors {
        counts[label as usize] += 1;
    }

    // Find the max count with a single pass — no Vec, no collect
    let max_count = counts.iter().copied().fold(0, u32::max);

    // Count how many labels share the max — if exactly one, return it directly
    let mut winner = 10u8; // sentinel: >9 means "tie so far"
    let mut unique = true;
    for (label, &c) in counts.iter().enumerate() {
        if c == max_count {
            if winner == 10 {
                winner = label as u8;
            } else {
                unique = false;
                break;
            }
        }
    }

    if unique { winner } else { neighbors[0].1 }
}
