// Library root — re-exports all public modules so binaries and tests can use
// `imgProcessing::<module>::<item>` without knowing the internal file layout.

pub mod benchmark;
pub mod data;
pub mod knn;
pub mod metrics;
pub mod parallel;
pub mod preprocess;
pub mod sequential;
