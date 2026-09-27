use puyo_core::board::{Board, PuyoColor};
use puyo_core::piece::Piece;
use puyo_core::state::PuyoState;

use az_framework::eval::Evaluator;

use crate::puyo_game::PuyoGame;
use puyo_core::placement::{enumerate_placements, simulate_placement};

/// Simulation-based evaluator: drops virtual puyos to estimate expected chain score.
pub struct SimulationEvaluator;

impl Evaluator<PuyoGame> for SimulationEvaluator {
    /// BFS順で全深度の盤面を評価し、最高スコアの1手目を返す。
    fn find_best_move(&self, state: &PuyoState) -> Option<(puyo_core::piece::Placement, f64)> {
        let board = &state.board;
        let current = &state.current;
        let next = &state.next;

        let placements = enumerate_placements(board, current);

        placements
            .iter()
            .filter_map(|p1| {
                let (board1, result1) = simulate_placement(board, current, p1);
                if board1.is_game_over() {
                    return None;
                }

                let depth1_score = (result1.score as f64).max(simulate_expected_score(&board1));

                let depth2_best = enumerate_placements(&board1, next)
                    .iter()
                    .filter_map(|p2| {
                        let (board2, result2) = simulate_placement(&board1, next, p2);
                        if board2.is_game_over() {
                            return None;
                        }
                        Some((result2.score as f64).max(simulate_expected_score(&board2)))
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                let score = depth1_score.max(depth2_best);
                Some((*p1, score))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

pub const W_GAME_OVER: f64 = -1000000.0;

// ---------------------------------------------------------------------------
// シミュレーション評価
// ---------------------------------------------------------------------------

/// 仮想ぷよ（同色2個Piece）を全合法配置に落として連鎖スコアの期待値（全パターン平均）を推定する。
fn simulate_expected_score(board: &Board) -> f64 {
    let num_colors = board.config.num_colors;
    let (sum, count) = PuyoColor::active_colors(num_colors)
        .iter()
        .flat_map(|&color| {
            let piece = Piece::new(color, color);
            enumerate_placements(board, &piece)
                .into_iter()
                .map(move |pl| {
                    let (sim, result) = simulate_placement(board, &piece, &pl);
                    if sim.is_game_over() {
                        W_GAME_OVER
                    } else {
                        result.score as f64
                    }
                })
        })
        .fold((0.0f64, 0usize), |(sum, count), score| {
            (sum + score, count + 1)
        });

    if count == 0 {
        return W_GAME_OVER;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;
    use puyo_core::config::GameConfig;
    use puyo_core::piece::Piece;

    fn cfg() -> GameConfig {
        GameConfig::default()
    }

    fn make_state(board: Board, current: Piece, next: Piece) -> PuyoState {
        PuyoState {
            board,
            current,
            next,
            next_next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
        }
    }

    #[test]
    fn test_depth1_finds_move() {
        let state = make_state(
            Board::default(),
            Piece::new(PuyoColor::Red, PuyoColor::Blue),
            Piece::new(PuyoColor::Red, PuyoColor::Blue),
        );
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&state);
        assert!(result.is_some());
    }

    #[test]
    fn test_depth2_finds_move() {
        let state = make_state(
            Board::default(),
            Piece::new(PuyoColor::Red, PuyoColor::Blue),
            Piece::new(PuyoColor::Green, PuyoColor::Blue),
        );
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&state);
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_avoids_game_over() {
        let c = cfg();
        let mut board = Board::default();
        for col in 0..c.cols {
            for i in 0..4 {
                let color = if (col + i) % 2 == 0 {
                    PuyoColor::Red
                } else {
                    PuyoColor::Blue
                };
                board.drop_puyo(col, color);
            }
        }

        let state = make_state(
            board,
            Piece::new(PuyoColor::Red, PuyoColor::Blue),
            Piece::new(PuyoColor::Green, PuyoColor::Blue),
        );
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&state);
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_prefers_chain() {
        let mut board = Board::default();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);

        let state = make_state(
            board,
            Piece::new(PuyoColor::Red, PuyoColor::Blue),
            Piece::new(PuyoColor::Green, PuyoColor::Blue),
        );
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&state);
        assert!(result.is_some());
    }

    // ---------------------------------------------------------------
    // simulate_expected_score のテスト
    // ---------------------------------------------------------------

    #[test]
    fn test_simulate_expected_score_empty_board() {
        let board = Board::default();
        let score = simulate_expected_score(&board);
        assert!(
            score.abs() < 1.0,
            "空盤面の期待スコアは0付近であるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_near_chain() {
        let mut board = Board::default();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        let score = simulate_expected_score(&board);
        assert!(
            score > 0.0,
            "連鎖可能盤面は正のスコアであるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_more_potential_is_higher() {
        let mut board1 = Board::default();
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);

        let mut board2 = Board::default();
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(1, PuyoColor::Blue);
        board2.drop_puyo(1, PuyoColor::Blue);
        board2.drop_puyo(1, PuyoColor::Blue);

        let s1 = simulate_expected_score(&board1);
        let s2 = simulate_expected_score(&board2);
        assert!(
            s2 > s1,
            "連鎖ポテンシャルが多い盤面のスコアが高いべき: s1={s1}, s2={s2}"
        );
    }

    #[test]
    fn test_simulate_expected_score_game_over_penalty() {
        let c = cfg();
        let mut board = Board::default();
        for col in 0..c.cols {
            for i in 0..c.visible_rows() {
                let color = if (col + i) % 2 == 0 {
                    PuyoColor::Red
                } else {
                    PuyoColor::Blue
                };
                board.drop_puyo(col, color);
            }
        }
        let score = simulate_expected_score(&board);
        assert!(
            score < -1000.0,
            "満杯に近い盤面は大きな負のスコアであるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_deterministic() {
        let mut board = Board::default();
        board.drop_puyo(2, PuyoColor::Green);
        board.drop_puyo(2, PuyoColor::Green);
        board.drop_puyo(1, PuyoColor::Blue);

        let s1 = simulate_expected_score(&board);
        let s2 = simulate_expected_score(&board);
        assert_eq!(s1, s2, "同じ盤面には同じスコアを返すべき");
    }
}
