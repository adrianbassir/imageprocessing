// CIFAR-10 binary parsing, dataset loading, train/test splitting.
// Each record: 1 label byte + 3072 pixel bytes (1024 R, 1024 G, 1024 B).
// This is the only module that reads files directly.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub struct Image {
    pub label: u8,       // class index 0–9 (airplane, automobile, …, truck)
    pub pixels: Vec<u8>, // 3072 raw bytes: 1024 R, 1024 G, 1024 B (channel-first)
}

pub struct Dataset {
    pub train: Vec<Image>, // images loaded from data_batch_1..5.bin (up to 50 000)
    pub test: Vec<Image>,  // images loaded from test_batch.bin (up to 10 000)
}

const RECORD_SIZE: usize = 1 + 3072; // 1 label byte + 3072 pixel bytes

/// Load all CIFAR-10 training batches and the test batch from `data_dir`.
pub fn load_dataset(data_dir: &str) -> Dataset {
    let train_files = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ];

    let mut train = Vec::new();
    for filename in &train_files {
        let path = Path::new(data_dir).join(filename);
        let batch = parse_batch(path.to_str().expect("invalid path"));
        train.extend(batch);
    }

    let test_path = Path::new(data_dir).join("test_batch.bin");
    let test = parse_batch(test_path.to_str().expect("invalid path"));

    Dataset { train, test }
}

/// Parse a single CIFAR-10 binary batch file into a list of Images.
fn parse_batch(path: &str) -> Vec<Image> {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("failed to open {}: {}", path, e));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .unwrap_or_else(|e: io::Error| panic!("failed to read {}: {}", path, e));

    assert!(
        buf.len() % RECORD_SIZE == 0,
        "unexpected file size in {}: {} bytes is not a multiple of {}",
        path,
        buf.len(),
        RECORD_SIZE
    );

    buf.chunks_exact(RECORD_SIZE)
        .map(|record| {
            let label = record[0];          // first byte is the class label
            let pixels = record[1..].to_vec(); // remaining 3072 bytes are pixel data
            Image { label, pixels }
        })
        .collect()
}
