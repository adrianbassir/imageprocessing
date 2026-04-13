// CIFAR-10 binary parsing, dataset loading, train/test splitting.
// Each record: 1 label byte + 3072 pixel bytes (1024 R, 1024 G, 1024 B).
// This is the only module that reads files directly.

pub struct Image {
    pub label: u8,
    pub pixels: Vec<u8>,
}

pub struct Dataset {
    pub train: Vec<Image>,
    pub test: Vec<Image>,
}

/// Load all CIFAR-10 training batches and the test batch from `data_dir`.
pub fn load_dataset(data_dir: &str) -> Dataset {
    todo!()
}

/// Parse a single CIFAR-10 binary batch file into a list of Images.
fn parse_batch(path: &str) -> Vec<Image> {
    todo!()
}
