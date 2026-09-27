pub mod eval;
#[cfg(feature = "nn")]
pub mod nn_eval;
pub mod puyo_game;

pub use puyo_core::placement;

// Re-export game-ai types for convenience
pub use az_framework::eval::Evaluator;

#[cfg(feature = "nn")]
pub use az_framework::nn_eval::{DirectInference, MctsConfig};
#[cfg(feature = "nn")]
pub use az_framework::{inference_server, mcts, model::GameModel};

#[cfg(feature = "nn")]
pub mod pvp_model;
#[cfg(feature = "nn")]
pub mod pvp_search;
