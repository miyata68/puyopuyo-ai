use crate::board::Board;
use crate::config::GameConfig;
use crate::piece::Piece;

/// MCTS/AI用の軽量ゲーム状態。
/// GameStateからUI関連（FallingPiece, phase等）を除いた純粋な盤面+ピースキュー。
#[derive(Clone)]
pub struct PuyoState {
    pub board: Board,
    pub current: Piece,
    pub next: Piece,
    pub next_next: Piece,
}

/// Board を [channel][row][col] のフラット f32 配列に変換する。
///
/// チャンネル:
/// - 0..(num_colors-1): 色ごとのone-hot
pub fn board_to_tensor_data(board: &Board) -> Vec<f32> {
    let cfg = &board.config;
    let cols = cfg.cols;
    let rows = cfg.rows;
    let tensor_size = cfg.tensor_size();

    let mut data = vec![0.0f32; tensor_size];

    for col in 0..cols {
        for row in 0..rows {
            let color = board.get(col, row);
            if color.is_color() {
                let ch = color as u8 as usize - 1;
                data[ch * rows * cols + row * cols + col] = 1.0;
            }
        }
    }

    data
}

/// 3ピース（current, next, next_next）をフラット f32 配列に変換する。
/// 各ピースの axis_color, satellite_color を num_colors 次元 one-hot でエンコード。
pub fn pieces_to_tensor_data(
    config: &GameConfig,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
) -> Vec<f32> {
    let num_colors = config.num_colors;
    let piece_tensor_size = config.piece_tensor_size();
    let mut data = vec![0.0f32; piece_tensor_size];
    let pieces = [current, next, next_next];
    for (i, piece) in pieces.iter().enumerate() {
        let base = i * (2 * num_colors);
        let axis_ch = piece.axis_color as u8 as usize - 1;
        data[base + axis_ch] = 1.0;
        let sat_ch = piece.satellite_color as u8 as usize - 1;
        data[base + num_colors + sat_ch] = 1.0;
    }
    data
}

/// コンテキストテンソル（FiLM条件付け用）に変換する。
pub fn context_to_tensor_data(
    config: &GameConfig,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
) -> Vec<f32> {
    pieces_to_tensor_data(config, current, next, next_next)
}
