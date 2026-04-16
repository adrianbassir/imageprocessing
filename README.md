# [PROJECT TITLE]
## _[YOUR NAME] & [PARTNER NAME]_
**[COURSE NAME]**

---

## Project Description

[PROJECT OVERVIEW — 2-3 sentences]

**Algorithm:** [ALGORITHM NAME]

**Dataset:** [DATASET NAME AND DESCRIPTION]

**Evaluation metrics:** [METRICS USED]

---

## Hardware Specifications

Benchmarks were run on two machines:

| | System 1 | System 2 |
|---|---|---|
| **Owner** | [SYSTEM 1 OWNER] | [SYSTEM 2 OWNER] |
| **CPU** | [SYSTEM 1 CPU] | [SYSTEM 2 CPU] |
| **Physical cores** | [SYSTEM 1 CORES] | [SYSTEM 2 CORES] |
| **RAM** | [SYSTEM 1 RAM] | [SYSTEM 2 RAM] |
| **OS** | [SYSTEM 1 OS] | [SYSTEM 2 OS] |

---

## Prerequisites

[PREREQUISITES — what needs to be installed and any system requirements]

---

## Setup Instructions

[SETUP STEPS — how to get the project running from scratch]

---

## Results

### Accuracy

| Implementation | Accuracy |
|---|---|
| Sequential | [ACCURACY]% |
| Threaded-4 | [ACCURACY]% |
| Rayon | [ACCURACY]% |

### Benchmark — System 1 ([SYSTEM 1 OWNER])

| Configuration | Threads | Time (ms) | Speedup | Efficiency |
|---|---|---|---|---|
| sequential | 1 | [TIME] | 1.000 | 1.000 |
| threaded-1 | 1 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-2 | 2 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-4 | 4 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| threaded-8 | 8 | [TIME] | [SPEEDUP] | [EFFICIENCY] |
| rayon | [THREADS] | [TIME] | [SPEEDUP] | [EFFICIENCY] |

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

[ANALYSIS — 3-5 sentences]

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto. CIFAR-10 dataset available at: https://www.cs.toronto.edu/~kriz/cifar.html
- The Rayon crate documentation: https://docs.rs/rayon
- Rust `std::thread` documentation: https://doc.rust-lang.org/std/thread/
