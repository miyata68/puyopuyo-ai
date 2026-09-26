use crate::board::{Board, ChainResult};
use crate::config::GameConfig;
use crate::piece::{FallingPiece, Orientation, Piece, Placement};
// use crate::rand::random_piece;
use crate::rand::{random_piece, seeded_piece};

/// Phase of the game state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    /// Player is controlling the falling piece.
    Falling,
    /// Piece has landed, resolving chains.
    Resolving,
    /// Game is over.
    GameOver,
}

impl GamePhase {
    /// Convert to integer representation (0=Falling, 1=Resolving, 2=GameOver).
    pub fn as_u8(&self) -> u8 {
        match self {
            GamePhase::Falling => 0,
            GamePhase::Resolving => 1,
            GamePhase::GameOver => 2,
        }
    }
}

/// Main game state.
#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Board,
    pub current_piece: Option<FallingPiece>,
    pub next_piece: Piece,
    pub next_next_piece: Piece,
    pub score: u32,
    pub max_chain: u32,
    pub phase: GamePhase,
    pub total_pieces: u32,

    fixed_seed: Option<u64>,
    next_piece_index: u64,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new(GameConfig::default())
    }
}

impl GameState {
    pub fn new(config: GameConfig) -> Self {
        let num_colors = config.num_colors;
        let current = random_piece(num_colors);
        let next = random_piece(num_colors);
        let next_next = random_piece(num_colors);

        let mut state = GameState {
            board: Board::new(&config),
            current_piece: None,
            next_piece: next,
            next_next_piece: next_next,
            score: 0,
            max_chain: 0,
            phase: GamePhase::Falling,
            total_pieces: 0,

            fixed_seed: None,
            next_piece_index: 0,
        };

        state.spawn_piece(current);
        state
    }

    pub fn new_with_seed(config: GameConfig, seed: u64) -> Self {
        let num_colors = config.num_colors;

        let current = seeded_piece(seed, 0, num_colors);
        let next = seeded_piece(seed, 1, num_colors);
        let next_next = seeded_piece(seed, 2, num_colors);

        let mut state = GameState {
            board: Board::new(&config),
            current_piece: None,
            next_piece: next,
            next_next_piece: next_next,
            score: 0,
            max_chain: 0,
            phase: GamePhase::Falling,
            total_pieces: 0,

            fixed_seed: Some(seed),

            // 0, 1, 2 はすでに生成済み
            next_piece_index: 3,
        };

        state.spawn_piece(current);
        state
    }

    /// Spawn a new piece at the top.
    fn spawn_piece(&mut self, piece: Piece) {
        self.current_piece = Some(FallingPiece::spawn(piece, &self.board.config));
    }

    /// Advance to the next piece.
    fn advance_piece(&mut self) {
        self.total_pieces += 1;
        let next = self.next_piece;
        self.next_piece = self.next_next_piece;
        // self.next_next_piece = random_piece(self.board.config.num_colors);
        self.next_next_piece = match self.fixed_seed {
            Some(seed) => {
                let piece = seeded_piece(seed, self.next_piece_index, self.board.config.num_colors);

                self.next_piece_index += 1;
                piece
            }

            None => random_piece(self.board.config.num_colors),
        };
        self.spawn_piece(next);
    }

    /// Move current piece left.
    pub fn move_left(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_move_left(&self.board)
        } else {
            false
        }
    }

    /// Move current piece right.
    pub fn move_right(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_move_right(&self.board)
        } else {
            false
        }
    }

    /// Rotate current piece clockwise.
    pub fn rotate_cw(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_rotate_cw(&self.board)
        } else {
            false
        }
    }

    /// Rotate current piece counter-clockwise.
    pub fn rotate_ccw(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_rotate_ccw(&self.board)
        } else {
            false
        }
    }

    /// Hard drop the current piece. Returns chain result if chains occurred.
    pub fn hard_drop(&mut self) -> Option<ChainResult> {
        if self.phase != GamePhase::Falling {
            return None;
        }

        let fp = self.current_piece.take()?;
        let placement = Placement::new(fp.col, fp.orientation);

        self.place_piece(&fp.piece, &placement);
        let result = self.resolve();
        Some(result)
    }

    /// Place a piece on the board at the given placement.
    pub fn place_piece(&mut self, piece: &Piece, placement: &Placement) {
        crate::placement::place_piece_on_board(&mut self.board, piece, placement);
    }

    /// Resolve chains after placing a piece.
    fn resolve(&mut self) -> ChainResult {
        self.phase = GamePhase::Resolving;

        let result = self.board.resolve_chains();
        self.score += result.score;
        if result.chain_count > self.max_chain {
            self.max_chain = result.chain_count;
        }

        // Check game over
        if self.board.is_game_over() {
            self.phase = GamePhase::GameOver;
        } else {
            self.phase = GamePhase::Falling;
            self.advance_piece();
        }

        result
    }

    /// Place a piece on the board without resolving chains or advancing to next piece.
    /// Used by the training loop to control chain resolution step by step.
    /// Returns true if a piece was placed, false if no current piece.
    pub fn place_piece_only(&mut self, placement: &Placement) -> bool {
        if let Some(fp) = self.current_piece.take() {
            self.place_piece(&fp.piece, placement);
            self.phase = GamePhase::Resolving;
            true
        } else {
            false
        }
    }

    /// Finalize game state after manual chain resolution.
    /// Updates score/max_chain, checks game over, advances to next piece.
    pub fn finalize_after_chains(&mut self, total_score: u32, max_chain: u32) {
        self.score += total_score;
        if max_chain > self.max_chain {
            self.max_chain = max_chain;
        }
        if self.board.is_game_over() {
            self.phase = GamePhase::GameOver;
        } else {
            self.phase = GamePhase::Falling;
            self.advance_piece();
        }
    }

    /// Apply a placement directly (used by AI). Returns chain result.
    pub fn apply_placement(&mut self, placement: &Placement) -> ChainResult {
        if let Some(fp) = self.current_piece.take() {
            self.place_piece(&fp.piece, placement);
            self.resolve()
        } else {
            ChainResult {
                chain_count: 0,
                score: 0,
            }
        }
    }

    /// Soft drop: move piece down by one step. Returns true if moved, false if landed.
    pub fn soft_drop(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            let axis_col = fp.col;
            let (dc, _dr) = fp.orientation.offset();
            let sat_col = (axis_col as i32 + dc).max(0) as usize;

            let axis_height = self.board.column_height(axis_col);
            let sat_height = self.board.column_height(sat_col);

            // The piece can't go below the highest column it occupies
            let min_row = match fp.orientation {
                Orientation::North => axis_height as f32, // axis on bottom
                Orientation::South => (axis_height + 1) as f32, // satellite lands at column_height, axis one above
                _ => axis_height.max(sat_height) as f32,        // side by side
            };

            if fp.row - 1.0 < min_row {
                // Would land
                false
            } else {
                fp.row -= 1.0;
                true
            }
        } else {
            false
        }
    }

    /// Tick: advance the game by one frame's worth of gravity.
    pub fn tick(&mut self, gravity: f32) -> Option<ChainResult> {
        if self.phase != GamePhase::Falling {
            return None;
        }

        let fp = self.current_piece.as_ref()?;
        let landing_row = self.landing_row_for(fp);
        let new_row = fp.row - gravity;

        if new_row <= landing_row {
            self.hard_drop()
        } else {
            self.current_piece.as_mut()?.row = new_row;
            None
        }
    }

    /// Compute the landing row for a falling piece based on current board state.
    fn landing_row_for(&self, fp: &FallingPiece) -> f32 {
        let axis_col = fp.col;
        let (dc, _) = fp.orientation.offset();
        let sat_col = (axis_col as i32 + dc).clamp(0, (self.board.config.cols - 1) as i32) as usize;

        let axis_height = self.board.column_height(axis_col);
        let sat_height = self.board.column_height(sat_col);

        match fp.orientation {
            Orientation::North => axis_height as f32,
            Orientation::South => (sat_height + 1) as f32,
            Orientation::East | Orientation::West => axis_height.max(sat_height) as f32,
        }
    }

    /// Restart the game.
    pub fn restart(&mut self) {
        let config = self.board.config;

        if let Some(seed) = self.fixed_seed {
            *self = GameState::new_with_seed(config, seed);
        } else {
            *self = GameState::new(config);
        }
    }

    pub fn restart_with_seed(&mut self, seed: u64) {
        let config = self.board.config;

        *self = GameState::new_with_seed(config, seed);
    }

    /// Get current piece info for rendering: (axis_color, sat_color, col, row, orientation_index)
    pub fn get_current_piece_info(&self) -> Option<(u8, u8, u8, f32, u8)> {
        self.current_piece.as_ref().map(|fp| {
            (
                fp.piece.axis_color as u8,
                fp.piece.satellite_color as u8,
                fp.col as u8,
                fp.row,
                fp.orientation.as_u8(),
            )
        })
    }

    /// Get next piece info: (axis_color, satellite_color)
    pub fn get_next_piece_info(&self) -> (u8, u8) {
        (
            self.next_piece.axis_color as u8,
            self.next_piece.satellite_color as u8,
        )
    }

    /// Get next-next piece info: (axis_color, satellite_color)
    pub fn get_next_next_piece_info(&self) -> (u8, u8) {
        (
            self.next_next_piece.axis_color as u8,
            self.next_next_piece.satellite_color as u8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor;

    fn cfg() -> GameConfig {
        GameConfig::default()
    }

    #[test]
    fn test_new_game() {
        let game = GameState::new(cfg());
        assert_eq!(game.phase, GamePhase::Falling);
        assert!(game.current_piece.is_some());
        assert_eq!(game.score, 0);
        assert_eq!(game.max_chain, 0);
    }

    #[test]
    fn test_hard_drop_and_next() {
        let mut game = GameState::new(cfg());
        game.hard_drop();
        if game.phase == GamePhase::Falling {
            assert!(game.current_piece.is_some());
        }
    }

    #[test]
    fn test_move_operations() {
        let c = cfg();
        let spawn_col = c.spawn_col();
        let mut game = GameState::new(c);
        assert_eq!(game.current_piece.as_ref().unwrap().col, spawn_col);

        game.move_left();
        assert_eq!(game.current_piece.as_ref().unwrap().col, spawn_col - 1);

        game.move_right();
        assert_eq!(game.current_piece.as_ref().unwrap().col, spawn_col);

        game.rotate_cw();
        assert_eq!(
            game.current_piece.as_ref().unwrap().orientation,
            Orientation::East
        );
    }

    #[test]
    fn test_restart() {
        let mut game = GameState::new(cfg());
        game.hard_drop();
        game.hard_drop();
        game.restart();
        assert_eq!(game.score, 0);
        assert_eq!(game.max_chain, 0);
        assert_eq!(game.phase, GamePhase::Falling);
    }

    #[test]
    fn test_soft_drop_south_orientation_landing() {
        let mut game = GameState::new(cfg());
        game.board.drop_puyo(2, PuyoColor::Red);
        game.board.drop_puyo(2, PuyoColor::Blue);
        game.board.drop_puyo(2, PuyoColor::Green);
        assert_eq!(game.board.column_height(2), 3);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Blue, PuyoColor::Red),
            col: 2,
            row: 7.0,
            orientation: Orientation::South,
        });

        while game.soft_drop() {}

        let fp = game.current_piece.as_ref().unwrap();
        assert_eq!(fp.row, 4.0);
    }

    #[test]
    fn test_tick_south_orientation_landing() {
        let mut game = GameState::new(cfg());
        game.board.drop_puyo(2, PuyoColor::Red);
        game.board.drop_puyo(2, PuyoColor::Blue);
        assert_eq!(game.board.column_height(2), 2);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Blue, PuyoColor::Green),
            col: 2,
            row: 4.0,
            orientation: Orientation::South,
        });
        game.phase = GamePhase::Falling;

        let _result = game.tick(1.5);

        assert_eq!(game.board.column_height(2), 4);
        assert_eq!(game.board.get(2, 2), PuyoColor::Green);
        assert_eq!(game.board.get(2, 3), PuyoColor::Blue);
    }

    #[test]
    fn test_south_orientation_empty_column() {
        let mut game = GameState::new(cfg());
        assert_eq!(game.board.column_height(2), 0);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            col: 2,
            row: 6.0,
            orientation: Orientation::South,
        });

        while game.soft_drop() {}

        let fp = game.current_piece.as_ref().unwrap();
        assert_eq!(fp.row, 1.0);
    }

    #[test]
    fn test_place_piece_north_defends_top_hidden_row_isolated() {
        let c = cfg();
        let rows = c.rows;
        let visible_rows = c.visible_rows();
        let mut game = GameState::new(c);
        for _ in 0..visible_rows {
            game.board.drop_puyo(0, PuyoColor::Red);
        }
        game.board.set(0, rows - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Blue, PuyoColor::Blue);
        let placement = Placement::new(0, Orientation::North);
        game.place_piece(&piece, &placement);

        assert_eq!(game.board.get(0, visible_rows), PuyoColor::Blue);
        assert_eq!(game.board.get(0, rows - 1), PuyoColor::Green);
    }
}
