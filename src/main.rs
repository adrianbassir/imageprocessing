// Entry point: CLI argument handling and benchmark orchestration.
// Loads the dataset, runs all three classifiers, and prints results.
//
// Usage:
//   cargo run --release [data_dir] [k] [train_limit] [test_limit]
//
// Defaults:
//   data_dir    = "data"
//   k           = 5
//   train_limit = 10000   (use 0 for full 50000)
//   test_limit  = 500     (use 0 for full 10000)

use imgProcessing::benchmark::run_all_benchmarks;
use imgProcessing::data::load_dataset;
use imgProcessing::metrics::{accuracy, print_accuracy, print_benchmark_table};
use imgProcessing::parallel::{classify_rayon, classify_threaded};
use imgProcessing::preprocess::normalize_all;
use imgProcessing::sequential::classify_sequential;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args.get(1).map(String::as_str).unwrap_or("data");
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

    // Apply limits for development / faster iteration
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
        "Using {} training images and {} test images (k={k}).",
        train_raw.len(),
        test_raw.len()
    );

    println!("Normalizing...");
    let train = normalize_all(train_raw);
    let test = normalize_all(test_raw);

    let ground_truth: Vec<u8> = test.iter().map(|img| img.label).collect();

    // --- Correctness check: verify all three implementations agree ---
    println!("\nRunning correctness check...");

    let seq_preds = classify_sequential(&train, &test, k);
    print_accuracy("sequential", accuracy(&seq_preds, &ground_truth));

    let train_arc = Arc::new(
        train
            .iter()
            .map(|img| imgProcessing::preprocess::NormalizedImage {
                label: img.label,
                features: img.features.clone(),
            })
            .collect::<Vec<_>>(),
    );

    let threaded_preds = classify_threaded(Arc::clone(&train_arc), &test, k, 4);
    print_accuracy("threaded-4", accuracy(&threaded_preds, &ground_truth));

    let rayon_preds = classify_rayon(&train, &test, k);
    print_accuracy("rayon", accuracy(&rayon_preds, &ground_truth));

    assert_eq!(
        seq_preds, threaded_preds,
        "sequential and threaded predictions differ"
    );
    assert_eq!(
        seq_preds, rayon_preds,
        "sequential and rayon predictions differ"
    );
    println!("All three implementations agree.");

    // --- Benchmarks ---
    println!("\nRunning benchmarks (3 runs each, median reported)...");
    let results = run_all_benchmarks(&train, &test, k);
    print_benchmark_table(&results);
}
