// Pixel normalization and feature vector construction.
// Converts raw u8 pixel values to normalized f32 feature vectors.
// Transformation is applied once at load time — not during classification.

use crate::data::Image;

pub struct NormalizedImage {
    pub label: u8,
    pub features: Vec<f32>,
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
