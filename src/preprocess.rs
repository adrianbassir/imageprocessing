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
    todo!()
}

/// Normalize an entire slice of images.
pub fn normalize_all(images: &[Image]) -> Vec<NormalizedImage> {
    todo!()
}
