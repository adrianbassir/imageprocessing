// Pixel normalization and feature vector construction.
// Converts raw u8 pixel values to normalized f32 feature vectors.
// Transformation is applied once at load time — not during classification.

use crate::data::Image;

pub struct NormalizedImage {
    pub label: u8,
    pub features: Vec<f32>,
}

/// Flat training data: all features packed into one contiguous Vec<f32>.
/// Eliminates per-image heap pointer chasing during distance computation,
/// giving the CPU prefetcher a single sequential stream to follow.
pub struct FlatTrainData {
    pub features: Vec<f32>, // row-major: [img0_f0..f3071, img1_f0..f3071, ...]
    pub labels: Vec<u8>,    // parallel array — labels[i] corresponds to features_of(i)
    pub dims: usize,        // number of features per image (3072 for CIFAR-10)
    pub n: usize,           // total number of training images
}

impl FlatTrainData {
    /// Return the feature slice for image at index `i`.
    #[inline(always)]
    pub fn features_of(&self, i: usize) -> &[f32] {
        let start = i * self.dims;
        &self.features[start..start + self.dims]
    }
}

/// Per-channel mean and standard deviation computed from the training set.
/// CIFAR-10 layout: 1024 R pixels, then 1024 G, then 1024 B (channel-first).
pub struct ChannelStats {
    pub mean: [f32; 3], // index 0=R, 1=G, 2=B; values in [0, 1] space
    pub std: [f32; 3],  // standard deviation in the same [0, 1] space
}

const PIXELS_PER_CHANNEL: usize = 1024;

/// Compute per-channel mean and std from raw training images.
/// Stats are computed in [0,1] space (i.e. after dividing by 255).
/// Both values are computed in a single two-pass scan to avoid fp drift
/// from a one-pass Welford update on 50 000 × 1024 = 51.2M samples.
pub fn compute_channel_stats(images: &[Image]) -> ChannelStats {
    let n = images.len();
    assert!(n > 0, "cannot compute stats from empty image set");

    // f64 accumulator avoids precision loss summing ~51M values (50 000 images × 1024 pixels)
    let total = (n * PIXELS_PER_CHANNEL) as f64;

    // --- Pass 1: mean ---
    let mut sum = [0.0f64; 3];
    for img in images {
        for c in 0..3 {
            let start = c * PIXELS_PER_CHANNEL;
            for &p in &img.pixels[start..start + PIXELS_PER_CHANNEL] {
                sum[c] += p as f64 / 255.0;
            }
        }
    }
    let mean = [
        (sum[0] / total) as f32,
        (sum[1] / total) as f32,
        (sum[2] / total) as f32,
    ];

    // --- Pass 2: variance ---
    let mut var = [0.0f64; 3];
    for img in images {
        for c in 0..3 {
            let start = c * PIXELS_PER_CHANNEL;
            for &p in &img.pixels[start..start + PIXELS_PER_CHANNEL] {
                let diff = (p as f64 / 255.0) - mean[c] as f64;
                var[c] += diff * diff;
            }
        }
    }
    let std = [
        (var[0] / total).sqrt() as f32,
        (var[1] / total).sqrt() as f32,
        (var[2] / total).sqrt() as f32,
    ];

    ChannelStats { mean, std }
}

/// Normalize a single image using per-channel z-score: (x - mean) / std.
/// CIFAR-10 pixel layout: indices [0..1024) = R, [1024..2048) = G, [2048..3072) = B.
pub fn normalize_zscore(image: &Image, stats: &ChannelStats) -> NormalizedImage {
    let mut features = Vec::with_capacity(image.pixels.len());
    for c in 0..3 {
        let start = c * PIXELS_PER_CHANNEL;
        for &p in &image.pixels[start..start + PIXELS_PER_CHANNEL] {
            let x = p as f32 / 255.0;
            features.push((x - stats.mean[c]) / stats.std[c]);
        }
    }
    NormalizedImage {
        label: image.label,
        features,
    }
}

/// Normalize a single image's pixels from [0, 255] to [0.0, 1.0].
pub fn normalize(image: &Image) -> NormalizedImage {
    let features = image.pixels.iter().map(|&p| p as f32 / 255.0).collect();
    NormalizedImage {
        label: image.label,
        features,
    }
}

/// Normalize an entire slice of images using per-channel z-score stats.
pub fn normalize_all_zscore(images: &[Image], stats: &ChannelStats) -> Vec<NormalizedImage> {
    images.iter().map(|img| normalize_zscore(img, stats)).collect()
}

/// Normalize an entire slice of images.
pub fn normalize_all(images: &[Image]) -> Vec<NormalizedImage> {
    images.iter().map(normalize).collect()
}

/// Pack normalized images into a flat contiguous layout for cache-efficient KNN.
pub fn flatten(images: &[NormalizedImage]) -> FlatTrainData {
    let n = images.len();
    let dims = if n > 0 { images[0].features.len() } else { 0 };
    let mut features = Vec::with_capacity(n * dims); // single allocation for all feature vectors
    let mut labels = Vec::with_capacity(n);
    for img in images {
        features.extend_from_slice(&img.features); // append this image's features contiguously
        labels.push(img.label);
    }
    FlatTrainData { features, labels, dims, n }
}
