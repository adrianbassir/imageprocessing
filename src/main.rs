// Entry point: CLI argument handling and benchmark orchestration.
// Loads the dataset, runs all three classifiers, and prints results.
//
// Usage:
//   cargo run --release [data_dir] [k] [train_limit] [test_limit]
//
// Defaults:
//   data_dir    = "data/cifar-10-batches-bin"
//   k           = 5
//   train_limit = 0  (full 50000)
//   test_limit  = 0  (full 10000)

use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};

use imgProcessing::benchmark::run_all_benchmarks;
use imgProcessing::data::load_dataset;
use imgProcessing::metrics::{accuracy, print_accuracy, print_benchmark_table};
use imgProcessing::parallel::{classify_rayon, classify_threaded};
use imgProcessing::preprocess::{flatten, normalize, NormalizedImage};
use imgProcessing::sequential::classify_sequential;

fn progress_bar(len: u64, prefix: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:<22} [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_prefix(prefix.to_string());
    pb
}

/// Classify all images in `test` using `f`, processing in chunks to drive a progress bar.
fn classify_with_bar<F>(label: &str, test: &[NormalizedImage], f: F) -> Vec<u8>
where
    F: Fn(&[NormalizedImage]) -> Vec<u8>,
{
    const CHUNK: usize = 500;
    let pb = progress_bar(test.len() as u64, label);
    let mut preds = Vec::with_capacity(test.len());
    for chunk in test.chunks(CHUNK) {
        preds.extend_from_slice(&f(chunk));
        pb.inc(chunk.len() as u64);
    }
    pb.finish();
    preds
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args.get(1).map(String::as_str).unwrap_or("data/cifar-10-batches-bin");
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let train_limit: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let test_limit: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    println!("Loading dataset from '{data_dir}'...");
    let dataset = load_dataset(data_dir);
    println!(
        "Loaded {} training images, {} test images.",
        dataset.train.len(),
        dataset.test.len()
    );

    let train_raw = if train_limit == 0 {
        &dataset.train[..]
    } else {
        &dataset.train[..train_limit.min(dataset.train.len())]
    };
    let test_raw = if test_limit == 0 {
        &dataset.test[..]
    } else {
        &dataset.test[..test_limit.min(dataset.test.len())]
    };

    println!(
        "Using {} training images and {} test images (k={k}).\n",
        train_raw.len(),
        test_raw.len()
    );

    // --- Normalize ---
    let pb = progress_bar(train_raw.len() as u64 + test_raw.len() as u64, "Normalizing");
    let train_normalized: Vec<NormalizedImage> = train_raw
        .iter()
        .map(|img| { pb.inc(1); normalize(img) })
        .collect();
    let test: Vec<NormalizedImage> = test_raw
        .iter()
        .map(|img| { pb.inc(1); normalize(img) })
        .collect();
    pb.finish();

    // --- Flatten training data for cache-efficient KNN ---
    let train = Arc::new(flatten(&train_normalized));
    println!();

    let ground_truth: Vec<u8> = test.iter().map(|img| img.label).collect();

    // --- Correctness check ---
    println!("Correctness check:");

    let seq_preds = classify_with_bar("  sequential", &test, |chunk| {
        classify_sequential(&train, chunk, k)
    });
    print_accuracy("  sequential", accuracy(&seq_preds, &ground_truth));

    let threaded_preds = classify_with_bar("  threaded-4", &test, |chunk| {
        classify_threaded(Arc::clone(&train), chunk, k, 4)
    });
    assert_eq!(seq_preds, threaded_preds, "sequential and threaded predictions differ");
    print_accuracy("  threaded-4", accuracy(&threaded_preds, &ground_truth));

    let rayon_preds = classify_with_bar("  rayon", &test, |chunk| {
        classify_rayon(&train, chunk, k)
    });
    assert_eq!(seq_preds, rayon_preds, "sequential and rayon predictions differ");
    print_accuracy("  rayon", accuracy(&rayon_preds, &ground_truth));

    println!("\nAll three implementations agree.\n");

    // --- Benchmarks ---
    println!("Running benchmarks (3 runs each, median reported)...");
    let results = run_all_benchmarks(&*train, &test, k);
    print_benchmark_table(&results);
}
