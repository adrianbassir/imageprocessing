// Timing infrastructure, speedup and efficiency calculations.
// Wraps classification runs with Instant timing; does not modify classification logic.
// All reported numbers must come from release-mode builds.

use std::sync::Arc;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

use crate::parallel::{classify_rayon, classify_threaded};
use crate::preprocess::NormalizedImage;
use crate::sequential::classify_sequential;

pub struct BenchmarkResult {
    pub label: String,
    pub num_threads: usize,
    pub elapsed: Duration,
    pub speedup: f64,
    pub efficiency: f64,
}

/// Time a single classification run and return elapsed duration.
pub fn time_run<F: FnOnce()>(f: F) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Compute speedup relative to the sequential baseline time.
/// speedup(N) = T(1) / T(N)
pub fn speedup(sequential_time: Duration, parallel_time: Duration) -> f64 {
    sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
}

/// Compute efficiency from speedup and thread count.
/// efficiency(N) = speedup(N) / N
pub fn efficiency(speedup: f64, num_threads: usize) -> f64 {
    speedup / num_threads as f64
}

const RUNS: usize = 3;
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

// Total number of timed runs: 1 sequential + 4 threaded + 1 rayon = 6 configs × RUNS each
const TOTAL_CONFIGS: u64 = (1 + 4 + 1) * RUNS as u64;

/// Run all configurations (sequential, threaded at 1/2/4/8, Rayon) and
/// collect BenchmarkResults. Each configuration is run RUNS times;
/// the median time is recorded.
pub fn run_all_benchmarks(
    train: &[NormalizedImage],
    test: &[NormalizedImage],
    k: usize,
) -> Vec<BenchmarkResult> {
    let pb = ProgressBar::new(TOTAL_CONFIGS);
    pb.set_style(
        ProgressStyle::with_template(
            "{prefix:<22} [{bar:40.cyan/blue}] {pos}/{len} runs  {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_prefix("Benchmarking");

    let mut results = Vec::new();

    // --- Sequential baseline ---
    let seq_time = median_time(RUNS, &pb, "sequential", || {
        classify_sequential(train, test, k);
    });
    results.push(BenchmarkResult {
        label: "sequential".to_string(),
        num_threads: 1,
        elapsed: seq_time,
        speedup: 1.0,
        efficiency: 1.0,
    });

    // Wrap in Arc once — no cloning during benchmark runs
    let train_arc = Arc::new(
        train
            .iter()
            .map(|img| NormalizedImage {
                label: img.label,
                features: img.features.clone(),
            })
            .collect::<Vec<_>>(),
    );
    let test_arc = Arc::new(
        test.iter()
            .map(|img| NormalizedImage {
                label: img.label,
                features: img.features.clone(),
            })
            .collect::<Vec<_>>(),
    );

    // --- Manual std::thread ---
    for &n in THREAD_COUNTS {
        let label = format!("threaded-{n}");
        let elapsed = median_time(RUNS, &pb, &label, || {
            classify_threaded(Arc::clone(&train_arc), Arc::clone(&test_arc), k, n);
        });
        let sp = speedup(seq_time, elapsed);
        let eff = efficiency(sp, n);
        results.push(BenchmarkResult {
            label,
            num_threads: n,
            elapsed,
            speedup: sp,
            efficiency: eff,
        });
    }

    // --- Rayon ---
    let rayon_elapsed = median_time(RUNS, &pb, "rayon", || {
        classify_rayon(train, test, k);
    });
    let sp = speedup(seq_time, rayon_elapsed);
    let rayon_threads = rayon::current_num_threads();
    let eff = efficiency(sp, rayon_threads);
    results.push(BenchmarkResult {
        label: "rayon".to_string(),
        num_threads: rayon_threads,
        elapsed: rayon_elapsed,
        speedup: sp,
        efficiency: eff,
    });

    pb.finish_with_message("done");
    results
}

/// Run `f` `n` times, ticking the progress bar each run, and return the median duration.
fn median_time<F: Fn()>(n: usize, pb: &ProgressBar, label: &str, f: F) -> Duration {
    let mut times: Vec<Duration> = (0..n)
        .map(|i| {
            pb.set_message(format!("{label} (run {}/{})", i + 1, n));
            let t = time_run(&f);
            pb.inc(1);
            t
        })
        .collect();
    times.sort();
    times[n / 2]
}
