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
    pub labels: Vec<u8>,
    pub dims: usize,
    pub n: usize,
}

impl FlatTrainData {
    /// Return the feature slice for image at index `i`.
    #[inline(always)]
    pub fn features_of(&self, i: usize) -> &[f32] {
        let start = i * self.dims;
        &self.features[start..start + self.dims]
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

/// Normalize an entire slice of images.
pub fn normalize_all(images: &[Image]) -> Vec<NormalizedImage> {
    images.iter().map(normalize).collect()
}

/// Pack normalized images into a flat contiguous layout for cache-efficient KNN.
pub fn flatten(images: &[NormalizedImage]) -> FlatTrainData {
    let n = images.len();
    let dims = if n > 0 { images[0].features.len() } else { 0 };
    let mut features = Vec::with_capacity(n * dims);
    let mut labels = Vec::with_capacity(n);
    for img in images {
        features.extend_from_slice(&img.features);
        labels.push(img.label);
    }
    FlatTrainData { features, labels, dims, n }
}
