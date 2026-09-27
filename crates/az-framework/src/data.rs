use serde::{Deserialize, Serialize};

/// A single AlphaZero self-play sample.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlphaZeroSample {
    /// Encoded board state.
    pub board_data: Vec<f32>,
    /// Encoded context.
    pub context_data: Vec<f32>,
    /// MCTS improved policy (num_actions floats, sums to 1.0).
    pub mcts_policy: Vec<f32>,
    /// Value target: discounted cumulative reward from this step.
    pub value_target: f32,
}

/// AlphaZero dataset of self-play samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlphaZeroDataset {
    pub samples: Vec<AlphaZeroSample>,
}

impl Default for AlphaZeroDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl AlphaZeroDataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: AlphaZeroDataset = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(dataset)
    }

    /// Load and merge multiple dataset files into one.
    pub fn load_multiple(paths: &[String]) -> std::io::Result<Self> {
        let mut merged = Self::new();
        for path in paths {
            let ds = Self::load(path)?;
            println!("  Loaded {} samples from {}", ds.samples.len(), path);
            merged.samples.extend(ds.samples);
        }
        Ok(merged)
    }
}
