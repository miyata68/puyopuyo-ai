//! Unaugmented, bounded-batch metrics on an inference-only network snapshot.
use crate::pvp_data::PvpAlphaZeroSample;
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use puyo_core::pvp_encoding::*;
use puyo_nn::pvp_model::PvpPuyoNet;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PolicyMetrics {
    pub samples: usize,
    pub policy_ce: f64,
    pub target_entropy: f64,
    pub policy_kl: f64,
    pub value_mse: f64,
}
impl std::fmt::Display for PolicyMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} policy_ce={:.6} target_entropy={:.6} policy_kl={:.6} value_mse={:.6}",
            self.samples, self.policy_ce, self.target_entropy, self.policy_kl, self.value_mse
        )
    }
}
/// Log-sum-exp over ALL legal actions, including legal zero-target actions.
/// Accumulate in f64, skipping p=0 terms (0 log 0 := 0).
pub fn metrics_from_predictions(
    logits: &[f32],
    values: &[f32],
    samples: &[PvpAlphaZeroSample],
) -> Result<PolicyMetrics, String> {
    if samples.is_empty()
        || logits.len() != samples.len() * NUM_ACTIONS
        || values.len() != samples.len()
    {
        return Err("invalid metric batch dimensions".into());
    }
    if logits.iter().chain(values).any(|v| !v.is_finite()) {
        return Err("non-finite evaluation prediction".into());
    }
    let mut metrics = PolicyMetrics {
        samples: samples.len(),
        ..Default::default()
    };
    for ((sample, row), value) in samples
        .iter()
        .zip(logits.as_chunks::<NUM_ACTIONS>().0)
        .zip(values)
    {
        sample.validate()?;
        let max = row
            .iter()
            .zip(&sample.legal_mask)
            .filter(|(_, legal)| **legal)
            .map(|(&v, _)| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = row
            .iter()
            .zip(&sample.legal_mask)
            .filter(|(_, legal)| **legal)
            .map(|(&v, _)| (v as f64 - max).exp())
            .sum::<f64>()
            .ln();
        for (&p, &logit) in sample.improved_policy.iter().zip(row) {
            if p > 0. {
                let p = p as f64;
                metrics.policy_ce -= p * (logit as f64 - max - log_sum);
                metrics.target_entropy -= p * p.ln();
            }
        }
        metrics.value_mse += (*value as f64 - sample.value_target as f64).powi(2);
    }
    let n = metrics.samples as f64;
    metrics.policy_ce /= n;
    metrics.target_entropy /= n;
    metrics.value_mse /= n;
    // Small negative residuals can arise from float32 target normalization.
    metrics.policy_kl = (metrics.policy_ce - metrics.target_entropy).max(0.);
    Ok(metrics)
}
pub fn forward_samples<B: Backend>(
    net: &PvpPuyoNet<B>,
    samples: &[PvpAlphaZeroSample],
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let board: Vec<_> = samples
        .iter()
        .flat_map(|s| s.board_data.iter().copied())
        .collect();
    let context: Vec<_> = samples
        .iter()
        .flat_map(|s| s.context_data.iter().copied())
        .collect();
    net.forward(
        Tensor::<B, 1>::from_floats(board.as_slice(), device).reshape([
            samples.len(),
            BOARD_CHANNELS,
            ROWS,
            COLS,
        ]),
        Tensor::<B, 1>::from_floats(context.as_slice(), device)
            .reshape([samples.len(), CONTEXT_SIZE]),
    )
}
/// Converts AD to the inner backend. No optimizer, backward or augmentation.
pub fn evaluate_snapshot<B: AutodiffBackend>(
    net: &PvpPuyoNet<B>,
    samples: &[PvpAlphaZeroSample],
    indices: &[usize],
    batch_size: usize,
    device: &B::Device,
) -> Result<PolicyMetrics, String> {
    if indices.is_empty() || batch_size == 0 || indices.iter().any(|&i| i >= samples.len()) {
        return Err("empty/invalid evaluation subset or batch size".into());
    }
    let snapshot = net.valid();
    let mut total = PolicyMetrics::default();
    for chunk in indices.chunks(batch_size) {
        let batch: Vec<_> = chunk.iter().map(|&i| samples[i].clone()).collect();
        let (logits, values) = forward_samples(&snapshot, &batch, device);
        let logits = logits
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        let values = values
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?;
        let m = metrics_from_predictions(&logits, &values, &batch)?;
        total.samples += m.samples;
        total.policy_ce += m.policy_ce * m.samples as f64;
        total.target_entropy += m.target_entropy * m.samples as f64;
        total.value_mse += m.value_mse * m.samples as f64;
    }
    total.policy_ce /= total.samples as f64;
    total.target_entropy /= total.samples as f64;
    total.value_mse /= total.samples as f64;
    total.policy_kl = (total.policy_ce - total.target_entropy).max(0.);
    Ok(total)
}
