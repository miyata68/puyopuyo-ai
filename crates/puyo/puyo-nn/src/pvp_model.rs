//! Separate PvP model and record type. Solo checkpoint layout is unchanged.
use crate::model::{ResidualBlock, ResidualBlockConfig};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;
use puyo_core::pvp_encoding::{BOARD_CHANNELS, COLS, CONTEXT_SIZE, NUM_ACTIONS, ROWS};
/// Dual-head CNN for Puyo Puyo with per-block FiLM conditioning (AlphaZero-style).
///
/// Architecture:
///   FiLM generator: context -> Linear -> ReLU -> Linear -> film_output
///     -> split into num_residual_blocks x (gamma[ch], beta[ch]) for each residual block
///   Backbone: stem (num_channels ch -> residual_channels ch) -> (BatchNorm + FiLM) ResidualBlock x N
///   Policy Head: Conv1x1 -> BatchNorm -> ReLU -> flatten -> FC(->num_actions)
///   Value Head:  Conv1x1 -> BatchNorm -> ReLU -> flatten -> FC(->64) -> ReLU -> FC(->1)
#[derive(Module, Debug)]
pub struct PvpPuyoNet<B: Backend> {
    // FiLM generator
    film_fc1: Linear<B>,
    film_fc2: Linear<B>,
    // CNN backbone
    stem: Conv2d<B>,
    res_blocks: Vec<ResidualBlock<B>>,
    // Policy head
    policy_conv: Conv2d<B>,
    policy_norm: BatchNorm<B>,
    policy_fc: Linear<B>,
    // Value head
    value_conv: Conv2d<B>,
    value_norm: BatchNorm<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
    activation: Relu,
}

/// Configuration for PvpPuyoNet. All architecture parameters are dynamic.
#[derive(Config, Debug)]
pub struct PvpPuyoNetConfig {
    /// Number of channels in residual blocks.
    #[config(default = 64)]
    pub residual_channels: usize,
    /// Number of residual blocks.
    #[config(default = 6)]
    pub num_residual_blocks: usize,
    /// Policy head Conv1x1 output channels.
    #[config(default = 2)]
    pub policy_conv_channels: usize,
    /// Value head Conv1x1 output channels.
    #[config(default = 1)]
    pub value_conv_channels: usize,
    /// Value head FC hidden size.
    #[config(default = 64)]
    pub value_hidden: usize,
    /// FiLM generator hidden layer size.
    #[config(default = 128)]
    pub film_hidden: usize,
}

impl PvpPuyoNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PvpPuyoNet<B> {
        let rc = self.residual_channels;
        let num_blocks = self.num_residual_blocks;
        let num_channels = BOARD_CHANNELS;
        let num_actions = NUM_ACTIONS;
        let context_size = CONTEXT_SIZE;
        let rows = ROWS;
        let cols = COLS;

        let pcc = self.policy_conv_channels;
        let vcc = self.value_conv_channels;
        let vh = self.value_hidden;
        let fh = self.film_hidden;

        let policy_flat = pcc * rows * cols;
        let value_flat = vcc * rows * cols;
        let film_output = rc * 2 * num_blocks;

        let res_block_config = ResidualBlockConfig::new(rc);
        let res_blocks = (0..num_blocks)
            .map(|_| res_block_config.init(device))
            .collect();

        PvpPuyoNet {
            film_fc1: LinearConfig::new(context_size, fh).init(device),
            film_fc2: LinearConfig::new(fh, film_output).init(device),
            stem: Conv2dConfig::new([num_channels, rc], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            res_blocks,
            // Policy head
            policy_conv: Conv2dConfig::new([rc, pcc], [1, 1]).init(device),
            policy_norm: BatchNormConfig::new(pcc).init(device),
            policy_fc: LinearConfig::new(policy_flat, num_actions).init(device),
            // Value head
            value_conv: Conv2dConfig::new([rc, vcc], [1, 1]).init(device),
            value_norm: BatchNormConfig::new(vcc).init(device),
            value_fc1: LinearConfig::new(value_flat, vh).init(device),
            value_fc2: LinearConfig::new(vh, 1).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PvpPuyoNet<B> {
    /// Forward pass.
    /// board: [batch, num_channels, rows, cols], context: [batch, context_tensor_size]
    /// Returns: (policy_logits [batch, num_actions], value [batch, 1])
    pub fn forward(
        &self,
        board: Tensor<B, 4>,
        context: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = board.dims()[0];

        // FiLM generator: context -> per-block (gamma, beta)
        let film = self.film_fc1.forward(context.clone());
        let film = self.activation.forward(film);
        let film = self.film_fc2.forward(film); // [batch, film_output]

        // Stem
        let mut x = self.stem.forward(board);
        x = self.activation.forward(x);

        // Residual blocks with per-block FiLM
        // Infer residual_channels and num_blocks from the film tensor dims
        let film_total = film.dims()[1];
        let num_blocks = self.res_blocks.len();
        let ch = film_total / (2 * num_blocks);
        for (i, block) in self.res_blocks.iter().enumerate() {
            let g_start = i * ch;
            let b_start = num_blocks * ch + i * ch;
            let gamma = film.clone().slice([0..batch_size, g_start..g_start + ch]);
            let beta = film.clone().slice([0..batch_size, b_start..b_start + ch]);
            x = block.forward(x, gamma, beta);
        }

        // Policy head: Conv1x1 -> BatchNorm -> ReLU -> flatten -> FC
        let p = self.policy_conv.forward(x.clone());
        let p = self.policy_norm.forward(p);
        let p = self.activation.forward(p);
        let policy_flat = p.dims()[1] * p.dims()[2] * p.dims()[3];
        let p = p.reshape([batch_size, policy_flat]);
        let policy_logits = self.policy_fc.forward(p);

        // Value head: Conv1x1 -> BatchNorm -> ReLU -> flatten -> FC -> ReLU -> FC
        let v = self.value_conv.forward(x);
        let v = self.value_norm.forward(v);
        let v = self.activation.forward(v);
        let value_flat = v.dims()[1] * v.dims()[2] * v.dims()[3];
        let v = v.reshape([batch_size, value_flat]);
        let v = self.value_fc1.forward(v);
        let v = self.activation.forward(v);
        let value = self.value_fc2.forward(v).tanh();

        (policy_logits, value)
    }
}
