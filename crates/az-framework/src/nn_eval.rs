use std::sync::Arc;

use burn::prelude::*;

use crate::mcts::{InferenceProvider, SeedProvider, TimeSeedProvider};
use crate::model::GameModel;

/// MCTS configuration (Gumbel AlphaZero).
pub struct MctsConfig {
    pub num_simulations: usize,
    /// Initial exploration constant for dynamic PUCT.
    /// c(s) = log((1 + N(s) + c_puct_base) / c_puct_base) + c_puct_init
    pub c_puct_init: f32,
    /// Base constant for dynamic PUCT (larger = more stable, less variation).
    pub c_puct_base: f32,
    /// Number of initial actions to sample via Gumbel-Top-k.
    pub m: usize,
    /// Q-value scaling for advantage computation.
    pub c_visit: f32,
    /// Discount factor for future rewards.
    pub gamma: f32,
    /// Number of leaves to evaluate per batch NN call (virtual loss).
    /// 1 = single inference per simulation (original behavior).
    /// >1 = batched inference with virtual loss for diverse path selection.
    pub num_leaves: usize,
    /// Seed provider for Gumbel noise generation.
    /// Default: `TimeSeedProvider` (time-based).
    pub seed_provider: Arc<dyn SeedProvider>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            c_puct_init: 1.5,
            c_puct_base: 19652.0,
            m: 16,
            c_visit: 5.0,
            gamma: 0.95,
            num_leaves: 1,
            seed_provider: Arc::new(TimeSeedProvider),
        }
    }
}

/// Direct (single-sample) inference using a burn backend.
/// Used for CPU inference (NdArray) in WASM and single-threaded scenarios.
pub struct DirectInference<B: Backend, M: GameModel<B>> {
    model: M,
    device: B::Device,
}

impl<B: Backend, M: GameModel<B>> DirectInference<B, M> {
    pub fn new(model: M, device: B::Device) -> Self {
        Self { model, device }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<B: Backend, M: GameModel<B> + Clone> Clone for DirectInference<B, M> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B: Backend, M: GameModel<B>> InferenceProvider for DirectInference<B, M> {
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32) {
        let (channels, rows, cols) = self.model.board_shape();
        let context_size = self.model.context_size();

        let board_tensor = Tensor::<B, 1>::from_floats(board_data, &self.device)
            .reshape([1, channels, rows, cols]);
        let context_tensor =
            Tensor::<B, 1>::from_floats(context_data, &self.device).reshape([1, context_size]);

        let (logits, value) = self.model.forward(board_tensor, context_tensor);

        let logits_vec = logits
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract logits tensor");
        let value_scalar = value
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract value tensor");
        let v_raw = if value_scalar.is_empty() {
            0.0
        } else {
            value_scalar[0]
        };
        let v = self.model.postprocess_value(v_raw);

        (logits_vec, v)
    }
}
