# Rust Image Classification Final Project
## _Adrian Bassir & Matthew Kane_
**CSI-380: Innov III; Emerging Languages**

---

## Project Description

This project is an implementation for the KNN (K nearest neighbors) algorithm in Rust using the CIFAR-10 dataset. We are comparing a single threaded sequentially run baseline with two parallel implementations of choice. The first parallel implementation is with std::thread partitioning and the second parallel implementation of choice is Rayon data-parallel iterators. The way that performance is monitored and evaluated is through classification accuracy, wall-clock execution time, parallel speedup as well as thread efficiency. This will be tested across two different machines in order to note any differences that may occur.

**Algorithm:** K Nearest Neighbors (KNN)

**Dataset:** CIFAR-10 (50k training images and 10k test images [32x32 RGB, 10 classes])

**Evaluation metrics:** Classification accuracy, execution time (ms), parallel speedup, thread efficiency

---

## Hardware Specifications

Benchmarks were run on two machines:

| | System 1 | System 2 |
|---|---|---|
| **Owner** | Adrian Bassir | [SYSTEM 2 OWNER] |
| **CPU** | Intel Core Ultra 7 255U | [SYSTEM 2 CPU] |
| **Physical cores** | 12 | [SYSTEM 2 CORES] |
| **RAM** | 32 GB DDR5-7600 | [SYSTEM 2 RAM] |
| **OS** | Windows 11 Home | [SYSTEM 2 OS] |

---

## Prerequisites

* Rust toolchain (rustc 1.75.0 or later)
* 16GB RAM recommended (more is fine too)
* 4 or more physical CPU cores (this is for meaningful speedup)
* External crates (managed by cargo): rayon, indicatif, criterion

Note: This project runs entirely on the CPU so no GPU is required 

---

## Setup Instructions

1. Make sure you are in the correct project directory. "cd ImageProcessing" will get you there.

2. Download the CIFAR-10 dataset from the official website. This repository will not include the dataset. Here is the official website: https://www.cs.toronto.edu/~kriz/cifar.html. From the website, go ahead and download the binary version (for C programs). It should download "cifar-10-binary.tar.gz". This will need to be extracted.

3. Once extracted, place the extracted CIFAR-10 folder into the data folder so it looks like "data/cifar-10-batches-bin/(bin files)

4. Run build command: "cargo build --release"

5. Run the program: "cargo run --release" (this will run the entire dataset with k=5)

6. Run tests: "cargo test"

7. Run criterion benchmarks: "cargo bench"

---

## Results

### Accuracy

| Implementation | Accuracy |
|---|---|
| Sequential | 35.69% |
| Threaded-4 | 35.69% |
| Rayon | 35.69% |

### Benchmark — System 1 (Adrian Bassir)

| Configuration | Threads | Time (ms) | Speedup | Efficiency |
|---|---|---|---|---|
| sequential | 1 | 112122 | 1.000 | 1.000 |
| threaded-1 | 1 | 100645 | 1.114 | 1.114 |
| threaded-2 | 2 | 58695 | 1.910 | 0.955 |
| threaded-4 | 4 | 39209 | 2.860 | 0.715 |
| threaded-8 | 8 | 29311 | 3.825 | 0.478 |
| rayon | 14 | 27762 | 4.039 | 0.288 |

### Benchmark — System 2 ([SYSTEM 2 OWNER])

| Configuration | Threads | Time (ms) | Speedup | Efficiency |
|---|---|---|---|---|
| sequential | 1 | [TIME] | 1.000 | 1.000 |
| threaded-1 | 1 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-2 | 2 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-4 | 4 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-8 | 8 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| rayon | [THREADS] | [TIME] | [SPEEDUP] | [EFFICIENCY] |

### Analysis

Speedup increases as more threads are added to the equation, however, this does not scale linearly as doubling threads does not double the speedup since efficiency does go down as more cores are added. This can be noticed as efficiency drops quite noticeably as we get into the higher core range going down to less than 0.5 of what we saw for a sequential run (as of threaded-8), an indicator that around half of CPU time is effectively wasted. Rayon shows an efficiency in the high 0.2 range which is an interesting result (marginally better than threaded-8 but also uses 14 threads instead of 8). This shows that returns significantly diminish past 8 cores.

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto. CIFAR-10 dataset available at: https://www.cs.toronto.edu/~kriz/cifar.html
- The Rayon crate documentation: https://docs.rs/rayon
- Rust `std::thread` documentation: https://doc.rust-lang.org/std/thread/
