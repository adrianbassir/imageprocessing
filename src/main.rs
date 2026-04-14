// Entry point: CLI argument handling and benchmark orchestration.
// Loads the dataset, runs all three classifiers, and prints results.
//
// Usage:
//   cargo run --release [data_dir] [k] [train_limit] [test_limit]
//
// Defaults (subset for development — use 0 for full dataset):
//   data_dir    = "data/cifar-10-batches-bin"
//   k           = 5
//   train_limit = 10000
//   test_limit  = 500

use std::sync::Arc;

use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};

use imgProcessing::benchmark::run_all_benchmarks;
use imgProcessing::data::load_dataset;
use imgProcessing::knn::classify;
use imgProcessing::metrics::{accuracy, print_accuracy, print_benchmark_table};
use imgProcessing::parallel::{classify_rayon, classify_threaded};
use imgProcessing::preprocess::NormalizedImage;

/// Run a parallel classifier closure, showing a spinner while it works.
fn run_with_spinner<T, F: FnOnce() -> T>(label: &str, f: F) -> T {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{prefix:<22} {spinner} {elapsed}")
            .unwrap()
            .tick_strings(&["-", "\\", "|", "/"]),
    );
    pb.set_prefix(label.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    let result = f();
    pb.with_finish(ProgressFinish::AndLeave).finish();
    result
}

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

/// Classify test images one at a time, ticking a progress bar after each.
fn classify_with_progress(
    label: &str,
    train: &[NormalizedImage],
    test: &[NormalizedImage],
    k: usize,
) -> Vec<u8> {
    let pb = progress_bar(test.len() as u64, label);
    let preds = test
        .iter()
        .map(|img| {
            let pred = classify(img, train, k);
            pb.inc(1);
            pred
        })
        .collect();
    pb.finish();
    preds
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args.get(1).map(String::as_str).unwrap_or("data/cifar-10-batches-bin");
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let train_limit: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let test_limit: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(500);

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
    let train: Vec<NormalizedImage> = train_raw
        .iter()
        .map(|img| { pb.inc(1); imgProcessing::preprocess::normalize(img) })
        .collect();
    let test: Vec<NormalizedImage> = test_raw
        .iter()
        .map(|img| { pb.inc(1); imgProcessing::preprocess::normalize(img) })
        .collect();
    pb.finish();
    println!();

    let ground_truth: Vec<u8> = test.iter().map(|img| img.label).collect();

    // Wrap in Arc once — shared across correctness check and benchmarks with no cloning
    let train_arc = Arc::new(train);
    let test_arc = Arc::new(test);

    // --- Correctness check ---
    println!("Correctness check:");

    let seq_preds = classify_with_progress("  sequential", &train_arc, &test_arc, k);
    print_accuracy("  sequential", accuracy(&seq_preds, &ground_truth));

    let threaded_preds = run_with_spinner("  threaded-4", || {
        classify_threaded(Arc::clone(&train_arc), Arc::clone(&test_arc), k, 4)
    });
    assert_eq!(seq_preds, threaded_preds, "sequential and threaded predictions differ");
    print_accuracy("  threaded-4", accuracy(&threaded_preds, &ground_truth));

    let rayon_preds = run_with_spinner("  rayon", || {
        classify_rayon(&train_arc, &test_arc, k)
    });
    assert_eq!(seq_preds, rayon_preds, "sequential and rayon predictions differ");
    print_accuracy("  rayon", accuracy(&rayon_preds, &ground_truth));

    println!("\nAll three implementations agree.\n");

    // --- Benchmarks ---
    println!("Running benchmarks (3 runs each, median reported)...");
    let results = run_all_benchmarks(&train_arc, &test_arc, k);
    print_benchmark_table(&results);
}
