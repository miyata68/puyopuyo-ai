//! Validated replay union with persistent, per-source sample split membership.
use crate::pvp_data::{augment_colors, PvpAlphaZeroDataset, PvpAlphaZeroSample};
use puyo_player::pvp_search::SearchRng;
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct ReplaySource {
    pub path: String,
    pub train: usize,
    pub validation: usize,
}
pub struct ReplayPool {
    train: Vec<PvpAlphaZeroSample>,
    validation: Vec<PvpAlphaZeroSample>,
    pub sources: Vec<ReplaySource>,
}
/// FNV-1a over UTF-8 path, delimiter, LE index and seed; SplitMix64 finalizer.
/// No process-randomized or platform-native hashing/byte order.
pub fn split_hash(source: &str, index: usize, seed: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for byte in source
        .as_bytes()
        .iter()
        .copied()
        .chain([0])
        .chain((index as u64).to_le_bytes())
        .chain(seed.to_le_bytes())
    {
        h = (h ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}
pub fn is_validation(source: &str, index: usize, seed: u64, fraction: f64) -> bool {
    (split_hash(source, index, seed) >> 11) as f64 / 9007199254740992. < fraction
}
impl ReplayPool {
    pub fn load(paths: &[String], fraction: f64, seed: u64) -> Result<Self, String> {
        if !fraction.is_finite() || fraction <= 0. || fraction >= 0.5 {
            return Err("validation-fraction must satisfy 0 < fraction < 0.5".into());
        }
        if paths.is_empty() {
            return Err("empty replay dataset list".into());
        }
        let mut seen = BTreeSet::new();
        let mut pool = Self {
            train: vec![],
            validation: vec![],
            sources: vec![],
        };
        for path in paths {
            let canonical = std::fs::canonicalize(path).map_err(|e| format!("{path}: {e}"))?;
            let source = canonical
                .to_str()
                .ok_or("dataset path must be UTF-8")?
                .replace('\\', "/");
            // Windows paths are case-insensitive. Also resolves relative paths and symlinks.
            let source = if cfg!(windows) {
                source.to_lowercase()
            } else {
                source
            };
            if !seen.insert(source.clone()) {
                return Err(format!("duplicate replay dataset path: {path}"));
            }
            let data = PvpAlphaZeroDataset::load(&canonical).map_err(|e| format!("{path}: {e}"))?;
            // load() calls validate() for metadata and every sample before any use.
            let mut counts = ReplaySource {
                path: source.clone(),
                train: 0,
                validation: 0,
            };
            for (index, sample) in data.samples.into_iter().enumerate() {
                if is_validation(&source, index, seed, fraction) {
                    pool.validation.push(sample);
                    counts.validation += 1;
                } else {
                    pool.train.push(sample);
                    counts.train += 1;
                }
            }
            pool.sources.push(counts);
        }
        if pool.train.is_empty() || pool.validation.is_empty() {
            return Err("replay split has empty train or validation set; add samples or change validation seed/fraction".into());
        }
        Ok(pool)
    }
    pub fn train(&self) -> &[PvpAlphaZeroSample] {
        &self.train
    }
    pub fn validation(&self) -> &[PvpAlphaZeroSample] {
        &self.validation
    }
    /// Only the training partition is reachable here; augmentation mutates a copy.
    pub fn training_batch(
        &self,
        size: usize,
        rng: &mut SearchRng,
        permutations: &[Vec<usize>],
    ) -> Result<Vec<PvpAlphaZeroSample>, String> {
        if size == 0 || permutations.is_empty() {
            return Err("empty training batch/permutations".into());
        }
        (0..size)
            .map(|_| {
                let mut sample = self.train[uniform_index(rng, self.train.len())].clone();
                augment_colors(
                    &mut sample,
                    &permutations[uniform_index(rng, permutations.len())],
                )?;
                Ok(sample)
            })
            .collect()
    }
}
/// Rejection sampling avoids modulo bias in the uniform replay distribution.
fn uniform_index(rng: &mut SearchRng, n: usize) -> usize {
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let x = rng.next_u64();
        if x >= threshold {
            return (x % n) as usize;
        }
    }
}
/// Fixed subset selected once, with an RNG separate from training/augmentation.
pub fn fixed_subset(len: usize, maximum: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<_> = (0..len).collect();
    let mut rng = SearchRng::new(seed);
    for i in (1..len).rev() {
        indices.swap(i, uniform_index(&mut rng, i + 1));
    }
    indices.truncate(maximum.min(len));
    indices.sort_unstable();
    indices
}
