use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::config::GameConfig;
use puyo_core::piece::Placement;
use puyo_core::state::{board_to_tensor_data, context_to_tensor_data, PuyoState};
use puyo_nn::model::PuyoNet;

use az_framework::eval::Evaluator;
use az_framework::mcts::{mcts_search, mcts_search_batched, InferenceProvider};
use az_framework::model::GameModel;
use az_framework::nn_eval::DirectInference;
pub use az_framework::nn_eval::MctsConfig;
use az_framework::value_transform::value_inverse_transform;

const VALUE_SCALE: f32 = 15.0;

use crate::puyo_game::PuyoGame;
use puyo_core::placement::{compute_valid_mask, index_to_placement};

type InferBackend = NdArray;

/// GameModel implementation wrapping PuyoNet for use with generic game-ai infrastructure.
pub struct PuyoGameModel<B: Backend> {
    pub net: PuyoNet<B>,
    game_config: GameConfig,
}

impl<B: Backend> PuyoGameModel<B> {
    pub fn new(net: PuyoNet<B>) -> Self {
        Self {
            net,
            game_config: GameConfig::default(),
        }
    }

    pub fn with_config(net: PuyoNet<B>, game_config: GameConfig) -> Self {
        Self { net, game_config }
    }
}

impl<B: Backend> Clone for PuyoGameModel<B> {
    fn clone(&self) -> Self {
        Self {
            net: self.net.clone(),
            game_config: self.game_config.clone(),
        }
    }
}

impl<B: Backend> GameModel<B> for PuyoGameModel<B> {
    fn board_shape(&self) -> (usize, usize, usize) {
        let gc = &self.game_config;
        (gc.num_channels(), gc.rows, gc.cols)
    }

    fn context_size(&self) -> usize {
        self.game_config.context_tensor_size()
    }

    fn num_actions(&self) -> usize {
        self.game_config.num_actions()
    }

    fn forward(&self, board: Tensor<B, 4>, context: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.net.forward(board, context)
    }

    fn postprocess_value(&self, raw: f32) -> f32 {
        value_inverse_transform(raw, VALUE_SCALE)
    }
}

/// Neural network evaluator using the dual-head PuyoNet.
/// Supports two modes: Policy-only (fast, for WASM) and MCTS (for training).
pub struct NnEvaluator {
    provider: DirectInference<InferBackend, PuyoGameModel<InferBackend>>,
    mcts_config: Option<MctsConfig>,
    game_config: GameConfig,
}

impl NnEvaluator {
    pub fn new(model: PuyoNet<InferBackend>, device: <InferBackend as Backend>::Device) -> Self {
        Self {
            provider: DirectInference::new(PuyoGameModel::new(model), device),
            mcts_config: None,
            game_config: GameConfig::default(),
        }
    }

    pub fn with_game_config(
        model: PuyoNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
        game_config: GameConfig,
    ) -> Self {
        Self {
            provider: DirectInference::new(
                PuyoGameModel::with_config(model, game_config.clone()),
                device,
            ),
            mcts_config: None,
            game_config,
        }
    }

    /// Enable MCTS mode for training.
    pub fn with_mcts(mut self, config: MctsConfig) -> Self {
        self.mcts_config = Some(config);
        self
    }

    /// Get access to the model (for MCTS in self-play).
    pub fn model(&self) -> &PuyoNet<InferBackend> {
        &self.provider.model().net
    }

    pub fn device(&self) -> &<InferBackend as Backend>::Device {
        self.provider.device()
    }
}

impl Evaluator<PuyoGame> for NnEvaluator {
    fn find_best_move(&self, state: &PuyoState) -> Option<(Placement, f64)> {
        let mask = compute_valid_mask(&state.board, &state.current);
        if !mask.iter().any(|&v| v) {
            return None;
        }

        let num_actions = self.game_config.num_actions();
        let cols = self.game_config.cols;

        // MCTS mode: use Gumbel tree search
        if let Some(ref mcts_config) = self.mcts_config {
            let (policy, q_values) = if mcts_config.num_leaves > 1 {
                mcts_search_batched::<PuyoGame>(state, &self.provider, mcts_config)
            } else {
                mcts_search::<PuyoGame>(state, &self.provider, mcts_config)
            };

            // Select the action with highest improved policy probability
            let mut best_index = 0;
            let mut best_prob = f64::NEG_INFINITY;
            for i in 0..num_actions {
                if mask[i] && (policy[i] as f64) > best_prob {
                    best_prob = policy[i] as f64;
                    best_index = i;
                }
            }
            // Return Q value (average cumulative reward) as the score
            return Some((
                index_to_placement(best_index, cols),
                q_values[best_index] as f64,
            ));
        }

        // Policy-only mode (fast, for WASM)
        let board_data = board_to_tensor_data(&state.board);
        let cfg = &state.board.config;
        let context_data =
            context_to_tensor_data(cfg, &state.current, &state.next, &state.next_next);

        let (logits_vec, _value) = self.provider.infer(&board_data, &context_data);

        // Masked argmax
        let mut best_index = 0;
        let mut best_logit = f64::NEG_INFINITY;
        for (i, &logit) in logits_vec.iter().enumerate() {
            if mask[i] && (logit as f64) > best_logit {
                best_logit = logit as f64;
                best_index = i;
            }
        }

        Some((index_to_placement(best_index, cols), best_logit))
    }

    fn set_num_simulations(&mut self, num_simulations: usize) {
        if let Some(ref mut config) = self.mcts_config {
            config.num_simulations = num_simulations;
        }
    }
}
