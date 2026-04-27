// Accuracy computation, results formatting, and output reporting.
// The only module that writes to stdout in structured form.

use crate::benchmark::BenchmarkResult;

/// Compute classification accuracy: correct predictions / total predictions.
pub fn accuracy(predicted: &[u8], ground_truth: &[u8]) -> f64 {
    assert_eq!(
        predicted.len(),
        ground_truth.len(),
        "predicted and ground_truth lengths must match"
    );
    let correct = predicted
        .iter()
        .zip(ground_truth.iter())
        .filter(|(p, g)| p == g)
        .count();
    correct as f64 / predicted.len() as f64
}

/// Print classification accuracy for a given run label.
pub fn print_accuracy(label: &str, acc: f64) {
    println!("{:<20} accuracy: {:.2}%", label, acc * 100.0);
}

/// Print a formatted benchmark results table to stdout.
pub fn print_benchmark_table(results: &[BenchmarkResult]) {
    println!(
        "\n{:<20} {:>8} {:>12} {:>10} {:>12}",
        "Configuration", "Threads", "Time (ms)", "Speedup", "Efficiency"
    );
    println!("{}", "-".repeat(66));
    for r in results {
        println!(
            "{:<20} {:>8} {:>12.1} {:>10.3} {:>12.3}",
            r.label,
            r.num_threads,
            r.elapsed.as_millis(),
            r.speedup,
            r.efficiency,
        );
    }
    println!();
}
