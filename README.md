# Sequential vs Parallel Image Classification in Rust
## _Adrian Bassir & Matthew Kane_
**CSI-380: Emerging Languages**

---

## Project Description

<!-- 2-3 sentence overview of what the project does. Include algorithm, dataset, and evaluation metrics. -->

_TODO: Write a brief overview of the project here._

**Algorithm:** <!-- e.g. K-Nearest Neighbors (KNN) -->

**Dataset:** <!-- e.g. CIFAR-10 — 50,000 training images, 10,000 test images, 32×32 RGB, 10 classes -->

**Evaluation metrics:** <!-- e.g. classification accuracy, execution time, speedup, efficiency -->

---

## Hardware Specifications

Benchmarks were run on two machines:

| | System 1 | System 2 |
|---|---|---|
| **Owner** | | |
| **CPU** | | |
| **Physical cores** | | |
| **RAM** | | |
| **OS** | | |

---

## Prerequisites

**Rust toolchain:** rustc 1.75.0 or later. Install via [rustup](https://rustup.rs/).

**System requirements:**
- RAM: 16 GB recommended (the full CIFAR-10 training set expands to ~600 MB of `f32` vectors in memory)
- CPU: 4+ physical cores recommended to observe meaningful parallel speedup at higher thread counts

**External crates** (declared in `Cargo.toml`):
- `rayon` — data-parallel iterators
- `indicatif` — progress bars
- `criterion` — benchmark harness (dev dependency)

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

By default this runs all three classifiers (sequential, threaded, Rayon) and prints a benchmark summary.

**Optional positional arguments:**

```
cargo run --release [data_dir] [k] [train_limit] [test_limit]
```

| Argument | Default | Description |
|---|---|---|
| `data_dir` | `data/cifar-10-batches-bin` | Path to the folder containing the `.bin` batch files |
| `k` | `5` | Number of nearest neighbors |
| `train_limit` | `0` (full 50,000) | Cap on training images; `0` means use all |
| `test_limit` | `0` (full 10,000) | Cap on test images; `0` means use all |

**Example — quick smoke test on a small subset:**

```bash
cargo run --release data/cifar-10-batches-bin 5 1000 200
```

### 6. Run tests

```bash
cargo test
```

### 7. Run benchmarks

```bash
cargo bench
```

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto. CIFAR-10 dataset available at: https://www.cs.toronto.edu/~kriz/cifar.html
- The Rayon crate documentation: https://docs.rs/rayon
- Rust `std::thread` documentation: https://doc.rust-lang.org/std/thread/
