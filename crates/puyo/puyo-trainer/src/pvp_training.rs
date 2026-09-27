//! Strict PvP model metadata and raw win/loss training losses.
use crate::pvp_data::{PvpAlphaZeroSample, PvpDataMetadata};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use puyo_core::pvp_encoding::*;
use puyo_nn::pvp_model::{PvpPuyoNet, PvpPuyoNetConfig};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvpModelMetadata {
    #[serde(flatten)]
    pub dimensions: PvpDataMetadata,
    pub num_colors: usize,
    #[serde(flatten)]
    pub architecture: PvpPuyoNetConfig,
}
impl PvpModelMetadata {
    pub fn new(architecture: PvpPuyoNetConfig) -> Self {
        Self {
            dimensions: Default::default(),
            num_colors: 4,
            architecture,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let a = &self.architecture;
        if self.dimensions != PvpDataMetadata::default() || self.num_colors != 4 {
            return Err("incompatible pvp model dimensions/version".into());
        }
        if [
            a.residual_channels,
            a.num_residual_blocks,
            a.policy_conv_channels,
            a.value_conv_channels,
            a.value_hidden,
            a.film_hidden,
        ]
        .contains(&0)
        {
            return Err("model dimensions must be positive".into());
        }
        Ok(())
    }
    pub fn read(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(format!("{path}.config.json"))
            .map_err(|e| format!("missing/invalid PvP metadata: {e}"))?;
        let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if value.get("model_kind").and_then(|v| v.as_str()) != Some("pvp") {
            return Err(
                "incompatible model kind: expected pvp (solo models are not supported)".into(),
            );
        }
        let meta: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        meta.validate()?;
        Ok(meta)
    }
    pub fn write(&self, path: &str) -> Result<(), String> {
        self.validate()?;
        std::fs::write(
            format!("{path}.config.json"),
            serde_json::to_string_pretty(self).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}
pub fn load_or_initialize<B: Backend>(
    path: &str,
    config: PvpPuyoNetConfig,
    seed: u64,
    device: &B::Device,
) -> Result<(PvpPuyoNet<B>, PvpModelMetadata), String> {
    let weights = std::path::Path::new(&format!("{path}.bin")).exists();
    let metadata = std::path::Path::new(&format!("{path}.config.json")).exists();
    if weights || metadata {
        let meta = PvpModelMetadata::read(path)?;
        if !weights {
            return Err("PvP metadata exists but weights are missing".into());
        }
        let net = meta
            .architecture
            .init::<B>(device)
            .load_file(
                path,
                &BinFileRecorder::<FullPrecisionSettings>::new(),
                device,
            )
            .map_err(|e| format!("PvP weight load failed: {e}"))?;
        Ok((net, meta))
    } else {
        let meta = PvpModelMetadata::new(config);
        meta.validate()?;
        B::seed(device, seed);
        println!("PvP model not found; using random initialization model_init_seed={seed}");
        Ok((meta.architecture.init(device), meta))
    }
}
pub fn save_model<B: Backend>(
    net: PvpPuyoNet<B>,
    meta: &PvpModelMetadata,
    path: &str,
) -> Result<(), String> {
    create_parent(path)?;
    net.save_file(path, &BinFileRecorder::<FullPrecisionSettings>::new())
        .map_err(|e| e.to_string())?;
    meta.write(path)
}
pub fn create_parent(path: &str) -> Result<(), String> {
    if let Some(p) = std::path::Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    Ok(())
}
/// Mask only genuinely illegal actions, not legal actions with zero visits.
pub fn losses<B: Backend>(
    logits: Tensor<B, 2>,
    values: Tensor<B, 2>,
    samples: &[PvpAlphaZeroSample],
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let n = samples.len();
    let target: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.improved_policy.iter().copied())
        .collect();
    let mask: Vec<f32> = samples
        .iter()
        .flat_map(|s| s.legal_mask.iter().map(|&v| if v { 0. } else { -1e9 }))
        .collect();
    let logits =
        logits + Tensor::<B, 1>::from_floats(mask.as_slice(), device).reshape([n, NUM_ACTIONS]);
    let shifted = logits.clone() - logits.max_dim(1);
    let logp = shifted.clone() - shifted.exp().sum_dim(1).log();
    let policy = (logp
        * Tensor::<B, 1>::from_floats(target.as_slice(), device).reshape([n, NUM_ACTIONS]))
    .sum()
    .neg()
        / n as f32;
    let z: Vec<f32> = samples.iter().map(|s| s.value_target).collect();
    let delta = values - Tensor::<B, 1>::from_floats(z.as_slice(), device).reshape([n, 1]);
    let value = (delta.clone() * delta).mean();
    (policy, value)
}
