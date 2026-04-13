# Sequential vs Parallel Image Classification in Rust
## _Adrian Bassir & Matthew kane_
**CSI-380: Emerging Languages**

**Authors:** Adrian Bassir, Matthew Kane

K-Nearest Neighbors classification on CIFAR-10, benchmarked across sequential and parallel execution models using `std::thread` and Rayon.

---

## Project Description

This project implements image classification using the K-Nearest Neighbors (KNN) algorithm on the CIFAR-10 dataset, then benchmarks sequential execution against two parallel strategies — manual `std::thread` and Rayon — across multiple thread counts and two hardware systems. The central objective is not classification accuracy: it is a rigorous, quantitative comparison of parallel execution strategies in Rust.

**Algorithm:** K-Nearest Neighbors (KNN) with Euclidean distance over normalized RGB feature vectors.

**Dataset:** CIFAR-10 — 50,000 training images and 10,000 test images at 32×32 pixels (3,072 features per image).

**Evaluation metrics:** classification accuracy, execution time, speedup, and efficiency across thread counts of 1, 2, 4, and 8.

---

## Prerequisites

**Rust toolchain:** rustc 1.75.0 or later. Install via [rustup](https://rustup.rs/).

**System requirements:**
- RAM: 16 GB recommended (the full CIFAR-10 training set expands to ~600 MB of `f32` vectors in memory)
- CPU: 4+ physical cores recommended to observe meaningful parallel speedup at higher thread counts

**External crates** (declared in `Cargo.toml`):
- `rayon` — data-parallel iterators
- Any additional utility crates (e.g., `byteorder` for binary parsing) as needed

No GPU, Python environment, or external build tooling is required.

---

## Setup Instructions

### 1. Open the project

```
cd image-processing
```

### 2. Download the CIFAR-10 dataset

The dataset is **not included** in this repository. Download it manually:

1. Go to: https://www.cs.toronto.edu/~kriz/cifar.html
2. Download **CIFAR-10 binary version (for C programs)** — `cifar-10-binary.tar.gz`
3. Extract the archive

### 3. Place dataset files

Copy the extracted binary files into the `data/` directory at the project root:

```
data/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
└── test_batch.bin
```

Do not rename the files. The loader expects these exact filenames.

### 4. Build

```bash
cargo build --release
```

### 5. Run

```bash
cargo run --release
```

By default this runs all three classifiers (sequential, threaded, Rayon) and prints a benchmark summary. Optional flags for subset size, thread count, and K value will be documented once the CLI interface is finalized.

### 6. Run tests

```bash
cargo test
```

### 7. Run benchmarks

```bash
cargo bench
```

---

## Why KNN

KNN was selected because it isolates the parallelization problem cleanly.

Most classification algorithms separate into a training phase and an inference phase. Training introduces gradient computation, parameter updates, or template construction — complexity that obscures the concurrency analysis. KNN has no training phase. Every classification query directly triggers the expensive computation: distance from the query image to all training images.

This structure creates a natural and honest comparison target. The sequential baseline does the same arithmetic as the parallel version — just without distribution. Speedup measurements reflect actual parallel gain, not algorithmic shortcuts.

KNN also has a straightforward performance model. The dominant cost is Euclidean distance computation over flat float vectors. That cost scales predictably with dataset size and feature dimensionality, which makes bottleneck analysis tractable and interpretable.

---

## Why CIFAR-10

CIFAR-10 was selected because it creates a meaningful computational workload without requiring specialized hardware or impractical runtimes.

Each CIFAR-10 image is 32x32 pixels with three RGB channels, producing a 3,072-element feature vector per image. The training set contains 50,000 images; the test set contains 10,000. A single classification query therefore requires computing 50,000 Euclidean distances over 3,072-dimensional vectors — roughly 150 million floating-point operations per query.

That scale matters for two reasons. First, it makes parallelization worthwhile: the per-query cost is high enough that distributing work across threads produces measurable gains. Second, RGB color data increases the feature vector size relative to grayscale alternatives, which amplifies the distance computation cost and gives parallelism more room to demonstrate speedup.

CIFAR-10 is also a standard benchmark with a well-documented binary format, which eliminates data wrangling as a project variable.

---

## Planned Architecture

The codebase will be organized around functional boundaries that map directly to the pipeline stages.

```
image-processing/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── src/
│   ├── main.rs            # Entry point, CLI argument handling, benchmark orchestration
│   ├── lib.rs             # Crate root, re-exports public API
│   ├── data.rs            # CIFAR-10 binary parsing, dataset loading, train/test splitting
│   ├── preprocess.rs      # Pixel normalization, feature vector construction
│   ├── knn.rs             # Core KNN logic: distance computation, neighbor selection, voting
│   ├── sequential.rs      # Single-threaded classification runner
│   ├── parallel.rs        # std::thread and Rayon classification runners
│   ├── benchmark.rs       # Timing infrastructure, speedup/efficiency calculations
│   └── metrics.rs         # Accuracy computation, results formatting, output reporting
├── data/
│   └── (see Setup Instructions — dataset files go here, not included in submission)
├── tests/
│   └── correctness.rs     # Integration tests validating all three classifiers agree
├── benchmarks/
│   └── knn_bench.rs       # Criterion or custom benchmark harness
└── others/
```

**Do not submit** the `target/` directory or any files from `data/`. The `.zip` or `.tar.gz` submission must contain only source files, `Cargo.toml`, `Cargo.lock`, and this README.

**`data.rs`** owns everything related to reading CIFAR-10 binary files. It produces typed structs (`Image`, `Dataset`) that the rest of the system consumes. No other module reads files directly.

**`preprocess.rs`** converts raw `u8` pixel values to normalized `f32` feature vectors. This transformation happens once at load time, not during classification.

**`knn.rs`** contains the distance function and neighbor aggregation logic. It is pure computation — no I/O, no threading. Both the sequential and parallel paths call into this module. This separation ensures correctness can be validated on the sequential path before parallelism is introduced.

**`sequential.rs`** contains `classify_sequential` — the single-threaded baseline. This is the reference implementation against which all parallel results are validated.

**`parallel.rs`** contains two entry points:
- `classify_threaded` — manual `std::thread` implementation
- `classify_rayon` — Rayon data-parallel implementation

All three classifiers accept the same inputs and produce the same output type. Behavioral equivalence is a correctness requirement, enforced by integration tests in `tests/`.

**`benchmark.rs`** wraps classification runs with timing and computes derived metrics. It does not modify classification logic.

**`metrics.rs`** computes accuracy and formats results for output. It is the only module that writes to stdout in structured form.

---

## Algorithm Workflow

### 1. Load Dataset

Parse CIFAR-10 binary files into structured records. Each record contains a label byte followed by 3,072 pixel bytes (1,024 red, 1,024 green, 1,024 blue, in channel-planar order). Store labels separately from pixel data.

### 2. Flatten and Normalize

Convert each image's pixel data into a flat `Vec<f32>`. Normalize pixel values from `[0, 255]` to `[0.0, 1.0]` by dividing by 255.0. This normalization is applied uniformly to both training and test images. The result is stored once per image at load time.

### 3. Compute Euclidean Distances

For a query image `q` and training image `t`, compute:

```
distance(q, t) = sqrt( sum( (q[i] - t[i])^2 ) for i in 0..3072 )
```

This is the core computation that parallelization targets. Each distance calculation is independent — there is no dependency between distance computations across training images.

### 4. Select K Nearest Neighbors

Collect distances paired with training labels. Sort or partially sort to find the K training images with smallest distance to the query. K is a runtime parameter, defaulting to a value validated against accuracy on the test subset.

### 5. Majority Vote

Among the K nearest neighbors, count label occurrences. Assign the query image the label with the highest count. In case of ties, use the label from the single nearest neighbor.

### 6. Evaluate

Compare predicted labels against ground-truth test labels. Compute classification accuracy as correct predictions divided by total test images.

---

## Parallelization Strategy

### Sequential Baseline

The sequential implementation is built first and treated as the reference. It classifies each test image one at a time, computing all 50,000 training distances per query in a single thread. This baseline establishes correct output and measures the cost that parallelism is expected to reduce.

The baseline must produce verified-correct accuracy before any parallel work begins. Parallel implementations are validated by comparing their outputs against the sequential baseline on the same inputs.

### Thread-Based Parallel Implementation (`std::thread`)

The test set is partitioned into `N` equal chunks, where `N` is the thread count. Each thread receives an immutable reference to the full training set (via `Arc`) and a slice of test images to classify. Threads operate independently with no synchronization during computation. Results are collected after all threads join.

This implementation exposes thread creation overhead, work partitioning behavior, and the cost of `Arc` reference counting under real workloads. It also demonstrates explicit ownership management, which is a core Rust concurrency concept.

Thread counts tested: 1, 2, 4, 8.

### Rayon-Based Parallel Implementation

The test image slice is processed using `par_iter()` from Rayon. Each test image's classification is an independent unit of work, making this a natural fit for Rayon's work-stealing scheduler. Rayon manages thread pool creation, work distribution, and load balancing automatically.

This implementation serves as a contrast to the manual threading approach. It demonstrates how data-parallel patterns in Rust can be expressed concisely while achieving comparable or better performance through adaptive scheduling.

---

## Concurrency Design Expectations

### Work Partitioning Philosophy

The parallelism target is the outer loop: classifying test images. Each test image classification is fully independent — no test image's result depends on another's. This is an embarrassingly parallel problem at the test-set level.

The inner loop — computing distance from one test image to all training images — is also parallelizable, but partitioning at the outer loop is preferred. It avoids inter-thread synchronization within a single classification and reduces scheduling overhead.

### Avoiding Shared Mutable State

Training data is read-only after load time. Preprocessing produces immutable feature vectors. No classification thread writes to shared state during computation. Results are accumulated after threads complete, not during.

Mutable state is confined to each thread's local scope during classification. The only shared data structure is the training set, which is shared immutably.

### When `Arc` Is Needed

The manual threading implementation requires `Arc<Vec<Image>>` to share the training dataset across threads without copying it. Rust's ownership model prevents sharing references across thread boundaries without explicit lifetime guarantees — `Arc` provides heap-allocated shared ownership with reference counting.

`Arc<Mutex<T>>` is not expected to be necessary anywhere in the core classification path. If a design choice requires it, that is a signal the work partitioning strategy needs revision.

### Minimizing Synchronization Overhead

Thread joining is the only synchronization point in the manual implementation. There are no channels, no mutexes, and no barriers during classification. Each thread writes to its own result buffer and returns it at join time.

For Rayon, synchronization is managed by the runtime. The `.collect()` at the end of a `par_iter()` chain is the effective synchronization point.

---

## Benchmarking Plan

### Measuring Execution Time

Use `std::time::Instant` to bracket classification runs. Timing begins immediately before the classification call and ends immediately after the last result is collected. Dataset loading and preprocessing are excluded from timing — they are infrastructure costs, not the target of analysis.

Each configuration (sequential, 2 threads, 4 threads, 8 threads, Rayon) is run multiple times and the median time is reported. This reduces variance from OS scheduling noise.

All benchmarks are run in release mode (`cargo run --release`). Debug-mode results will not be reported.

### Speedup

```
speedup(N) = T(1) / T(N)
```

Where `T(1)` is sequential execution time and `T(N)` is parallel execution time with N threads. Speedup is computed relative to the single-threaded sequential baseline, not the single-thread parallel configuration.

### Efficiency

```
efficiency(N) = speedup(N) / N
```

Efficiency measures how well additional threads are utilized. An efficiency of 1.0 means perfect linear scaling. Values below 1.0 reflect overhead, contention, or memory bandwidth saturation. Efficiency is expected to decrease as thread count increases — the question is how quickly.

### Thread Counts

Benchmarks run at: 1, 2, 4, 8 threads.

Results are presented in a table comparing execution time, speedup, and efficiency across these counts for both the manual threading and Rayon implementations.

---

## Evaluation Expectations

The final output should demonstrate three things:

**Correctness.** The sequential, threaded, and Rayon implementations must produce identical classification results on the same inputs. Accuracy is measured on the test set and reported alongside performance metrics. The accuracy value itself is expected to be moderate — KNN on raw pixel values is not a high-accuracy classifier — but it must be consistent across all implementations.

**Performance Differences.** The parallel implementations should show meaningful speedup over the sequential baseline, particularly at 4 and 8 threads. The magnitude and shape of speedup curves will reflect real costs: thread creation overhead, memory bandwidth contention, and Rayon's scheduling efficiency.

**Scalability Trends.** Efficiency numbers are expected to degrade as thread count increases. The benchmark should show where that degradation becomes significant and what it implies about the bottleneck — whether it is compute-bound, memory-bound, or overhead-bound.

**Cross-Hardware Comparison.** The full benchmark suite must be run on two distinct hardware systems. Results from both systems — execution time, speedup, and efficiency tables — will be compared in the final presentation. Hardware specifications (CPU model, core count, RAM) must be recorded for both machines so that performance differences can be interpreted in context. This is a graded requirement.

---

## Expected Bottlenecks

### Memory Bandwidth

CIFAR-10's training set occupies roughly 150 MB of `f32` feature vectors (50,000 images × 3,072 floats × 4 bytes). Classifying a single test image requires reading all 150 MB to compute distances. Multiple threads doing this simultaneously contend for memory bus bandwidth. This is likely the primary constraint on scaling efficiency beyond 4 threads on most development hardware.

### Thread Creation Overhead

Spawning threads with `std::thread::spawn` has measurable latency. For small test subsets used during development, this overhead can dominate execution time and make parallel implementations appear slower than sequential. Thread creation cost should be discussed explicitly in the final benchmarks and contextualized against the full-dataset workload.

### Distance Computation Cost Per Image

At 3,072 dimensions, each Euclidean distance computation is non-trivial but small. The cost is in the aggregate: 50,000 distances per test image, across however many test images are classified. Individual distance functions are not worth optimizing in isolation — the bottleneck is throughput across the full loop.

### CIFAR-10 Data Size

Parsing and preprocessing 50,000 training images takes real time. This cost is paid once at startup and excluded from benchmarks. However, it informs the decision to develop and validate on dataset subsets: loading a full dataset for each benchmark iteration would make iteration slow and frustrating.

---

## Implementation Milestones

### Phase 1: Sequential Correctness

Implement data loading, preprocessing, and KNN classification sequentially. Validate that the system produces reasonable accuracy on a small test subset (e.g., 100 test images against 1,000 training images). Get the pipeline right before scaling it.

### Phase 2: Subset Validation

Run the sequential implementation on a larger subset (e.g., 1,000 test images, 10,000 training images). Confirm accuracy is stable and timing is measurable. Establish the sequential baseline numbers that parallel implementations will be compared against.

### Phase 3: Thread-Based Parallelization

Implement the manual `std::thread` classifier. Validate output against the sequential baseline. Run benchmarks at 1, 2, 4, 8 threads and record results.

### Phase 4: Rayon Parallelization

Implement the Rayon classifier. Validate output. Run benchmarks and record results alongside the manual threading numbers.

### Phase 5: Benchmarking and Analysis

Run full benchmarks in release mode. Compute speedup and efficiency tables. Identify where scaling breaks down and form an explanation grounded in the bottlenecks described above.

### Phase 6: Refinement and Reporting

Clean up output formatting, ensure metrics are clearly reported, and write the final analysis. Confirm that all three classifiers agree on predictions. Finalize code for submission.

---

## Practical Constraints

**Start with subsets.** Full CIFAR-10 classification (10,000 test images against 50,000 training images) is expensive enough to make development iteration slow. Initial development should use 100-1,000 test images and 1,000-10,000 training images. Scale to the full dataset only for final benchmarks.

**Correctness before optimization.** The sequential implementation must be verified correct before any parallel work begins. Parallel implementations must be validated against the sequential baseline before benchmarking. Incorrect fast code is not useful output.

**Benchmark in release mode.** All reported timing numbers must come from `cargo run --release` or `cargo bench`. Debug builds include bounds checks, disabled inlining, and no auto-vectorization — they do not reflect the algorithm's actual performance. Development can use debug mode; benchmarks cannot.

**K is a parameter, not a constant.** The number of neighbors K should be configurable at runtime. A reasonable default (e.g., K=5 or K=7) should be selected based on accuracy validation on the subset, not assumed.

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto. CIFAR-10 dataset available at: https://www.cs.toronto.edu/~kriz/cifar.html
- The Rayon crate documentation: https://docs.rs/rayon
- Rust `std::thread` documentation: https://doc.rust-lang.org/std/thread/
