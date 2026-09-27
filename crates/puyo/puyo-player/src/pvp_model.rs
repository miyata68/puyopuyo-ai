use az_framework::model::GameModel;
use burn::prelude::*;
use puyo_core::pvp_encoding::*;
use puyo_nn::pvp_model::PvpPuyoNet;
#[derive(Clone)]
pub struct PvpGameModel<B: Backend> {
    pub net: PvpPuyoNet<B>,
}
impl<B: Backend> GameModel<B> for PvpGameModel<B> {
    fn board_shape(&self) -> (usize, usize, usize) {
        (BOARD_CHANNELS, ROWS, COLS)
    }
    fn context_size(&self) -> usize {
        CONTEXT_SIZE
    }
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }
    fn forward(&self, b: Tensor<B, 4>, c: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.net.forward(b, c)
    }
    fn postprocess_value(&self, raw: f32) -> f32 {
        raw
    }
}
