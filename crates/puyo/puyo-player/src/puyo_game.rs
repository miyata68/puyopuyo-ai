use std::sync::OnceLock;

use az_framework::game::Game;
use puyo_core::board::ChainResult;
use puyo_core::config::GameConfig;
use puyo_core::piece::Placement;
use puyo_core::state::{board_to_tensor_data, context_to_tensor_data, PuyoState};

use puyo_core::placement::{
    compute_valid_mask, enumerate_placements, index_to_placement, placement_to_index,
    simulate_placement,
};

/// グローバルゲーム設定。`init_config()` で一度だけ設定可能。
static GAME_CONFIG: OnceLock<GameConfig> = OnceLock::new();

/// ランタイムのゲーム設定を初期化する。
/// 最初の呼び出しのみ有効。2回目以降は無視される。
/// 呼ばない場合は `GameConfig::default()` が使われる。
pub fn init_config(config: GameConfig) {
    let _ = GAME_CONFIG.set(config);
}

/// ぷよぷよゲームの Game trait 実装。
#[derive(Clone)]
pub struct PuyoGame;

impl PuyoGame {
    fn config() -> GameConfig {
        GAME_CONFIG.get().copied().unwrap_or_default()
    }
}

impl Game for PuyoGame {
    type State = PuyoState;
    type Action = Placement;
    type ActionResult = ChainResult;

    fn num_actions() -> usize {
        Self::config().num_actions()
    }

    fn board_tensor_shape() -> (usize, usize, usize) {
        let gc = Self::config();
        (gc.num_channels(), gc.rows, gc.cols)
    }

    fn context_tensor_size() -> usize {
        Self::config().context_tensor_size()
    }

    fn is_terminal(state: &PuyoState) -> bool {
        state.board.is_game_over()
    }

    fn legal_actions(state: &PuyoState) -> Vec<Placement> {
        enumerate_placements(&state.board, &state.current)
    }

    fn valid_action_mask(state: &PuyoState) -> Vec<bool> {
        compute_valid_mask(&state.board, &state.current)
    }

    fn action_to_index(action: &Placement) -> usize {
        placement_to_index(action)
    }

    fn index_to_action(index: usize) -> Placement {
        let gc = Self::config();
        index_to_placement(index, gc.cols)
    }

    fn apply_action(state: &PuyoState, action: &Placement) -> (PuyoState, ChainResult) {
        let (new_board, chain_result) = simulate_placement(&state.board, &state.current, action);
        let new_state = PuyoState {
            board: new_board,
            current: state.next,
            next: state.next_next,
            // next_next は advance_turn で設定される。暫定的に next_next をコピー。
            next_next: state.next_next,
        };
        (new_state, chain_result)
    }

    fn reward(result: &ChainResult) -> f32 {
        result.score as f32
    }

    fn encode_board(state: &PuyoState) -> Vec<f32> {
        board_to_tensor_data(&state.board)
    }

    fn encode_context(state: &PuyoState) -> Vec<f32> {
        let cfg = &state.board.config;
        context_to_tensor_data(cfg, &state.current, &state.next, &state.next_next)
    }

    fn advance_turn(state: &mut PuyoState) {
        let num_colors = state.board.config.num_colors;
        state.next_next = puyo_core::rand::random_piece(num_colors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::{Board, PuyoColor};
    use puyo_core::piece::{Orientation, Piece};

    #[test]
    fn test_puyo_game_num_actions() {
        let gc = GameConfig::default();
        assert_eq!(PuyoGame::num_actions(), gc.cols * 4);
    }

    #[test]
    fn test_puyo_game_apply_action() {
        let state = PuyoState {
            board: Board::default(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let action = Placement::new(0, Orientation::North);
        let (new_state, result) = PuyoGame::apply_action(&state, &action);
        assert!(!PuyoGame::is_terminal(&new_state));
        assert_eq!(result.score, 0);
        // current が next に進んでいること
        assert_eq!(new_state.current, state.next);
    }

    #[test]
    fn test_puyo_game_encode_board_size() {
        let state = PuyoState {
            board: Board::default(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let tensor = PuyoGame::encode_board(&state);
        let (ch, h, w) = PuyoGame::board_tensor_shape();
        assert_eq!(tensor.len(), ch * h * w);
    }

    #[test]
    fn test_puyo_game_encode_context_size() {
        let state = PuyoState {
            board: Board::default(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let tensor = PuyoGame::encode_context(&state);
        assert_eq!(tensor.len(), PuyoGame::context_tensor_size());
    }

    #[test]
    fn test_random_piece_valid() {
        let gc = GameConfig::default();
        let p = puyo_core::rand::random_piece(gc.num_colors);
        assert_ne!(p.axis_color, PuyoColor::Empty);
        assert_ne!(p.satellite_color, PuyoColor::Empty);
    }
}
