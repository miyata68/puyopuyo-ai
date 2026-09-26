use crate::board::{Board, PuyoColor};
use crate::config::GameConfig;

/// Orientation of the satellite puyo relative to the axis puyo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    North, // satellite above axis
    East,  // satellite right of axis
    South, // satellite below axis
    West,  // satellite left of axis
}

impl Orientation {
    /// Rotate clockwise.
    pub fn rotate_cw(self) -> Self {
        match self {
            Orientation::North => Orientation::East,
            Orientation::East => Orientation::South,
            Orientation::South => Orientation::West,
            Orientation::West => Orientation::North,
        }
    }

    /// Rotate counter-clockwise.
    pub fn rotate_ccw(self) -> Self {
        match self {
            Orientation::North => Orientation::West,
            Orientation::West => Orientation::South,
            Orientation::South => Orientation::East,
            Orientation::East => Orientation::North,
        }
    }

    /// Get the (dcol, drow) offset for the satellite relative to axis.
    pub fn offset(self) -> (i32, i32) {
        match self {
            Orientation::North => (0, 1),
            Orientation::East => (1, 0),
            Orientation::South => (0, -1),
            Orientation::West => (-1, 0),
        }
    }

    /// Convert to integer representation (0=North, 1=East, 2=South, 3=West).
    pub fn as_u8(self) -> u8 {
        match self {
            Orientation::North => 0,
            Orientation::East => 1,
            Orientation::South => 2,
            Orientation::West => 3,
        }
    }
}

/// A two-puyo piece (tsumo). axis_color is the pivot, satellite_color orbits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub axis_color: PuyoColor,
    pub satellite_color: PuyoColor,
}

impl Piece {
    pub fn new(axis: PuyoColor, satellite: PuyoColor) -> Self {
        Piece {
            axis_color: axis,
            satellite_color: satellite,
        }
    }
}

/// A specific placement: which column the axis lands in and the orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Placement {
    pub col: usize, // axis column (0..COLS-1)
    pub orientation: Orientation,
}

impl Placement {
    pub fn new(col: usize, orientation: Orientation) -> Self {
        Placement { col, orientation }
    }

    /// Get the satellite column given this placement.
    /// Returns None if out of bounds.
    pub fn satellite_col(&self, cols: usize) -> Option<usize> {
        let (dc, _) = self.orientation.offset();
        let sc = self.col as i32 + dc;
        if !(0..cols as i32).contains(&sc) {
            None
        } else {
            Some(sc as usize)
        }
    }
}

/// The falling piece state during gameplay.
#[derive(Debug, Clone)]
pub struct FallingPiece {
    pub piece: Piece,
    pub col: usize, // axis column
    pub row: f32,   // axis row (fractional for smooth fall)
    pub orientation: Orientation,
}

impl FallingPiece {
    pub fn spawn(piece: Piece, config: &GameConfig) -> Self {
        FallingPiece {
            piece,
            col: config.spawn_col(),
            row: config.visible_rows() as f32,
            orientation: Orientation::North,
        }
    }

    /// Get the satellite position (col, row_offset).
    pub fn satellite_pos(&self) -> (i32, i32) {
        let (dc, dr) = self.orientation.offset();
        (self.col as i32 + dc, self.row as i32 + dr)
    }

    /// Try to move left. Returns true if successful.
    pub fn try_move_left(&mut self, board: &Board) -> bool {
        let new_col = self.col as i32 - 1;
        if self.can_occupy(new_col, self.row as i32, self.orientation, board) {
            self.col = new_col as usize;
            true
        } else {
            false
        }
    }

    /// Try to move right. Returns true if successful.
    pub fn try_move_right(&mut self, board: &Board) -> bool {
        let new_col = self.col as i32 + 1;
        if self.can_occupy(new_col, self.row as i32, self.orientation, board) {
            self.col = new_col as usize;
            true
        } else {
            false
        }
    }

    /// Try to rotate clockwise with wall kick.
    pub fn try_rotate_cw(&mut self, board: &Board) -> bool {
        let new_ori = self.orientation.rotate_cw();
        // Try normal rotation
        if self.can_occupy(self.col as i32, self.row as i32, new_ori, board) {
            self.orientation = new_ori;
            return true;
        }
        // Wall kick: try shifting opposite to satellite direction
        let (dc, _) = new_ori.offset();
        let kick_col = self.col as i32 - dc;
        if self.can_occupy(kick_col, self.row as i32, new_ori, board) {
            self.col = kick_col as usize;
            self.orientation = new_ori;
            return true;
        }
        false
    }

    /// Try to rotate counter-clockwise with wall kick.
    pub fn try_rotate_ccw(&mut self, board: &Board) -> bool {
        let new_ori = self.orientation.rotate_ccw();
        if self.can_occupy(self.col as i32, self.row as i32, new_ori, board) {
            self.orientation = new_ori;
            return true;
        }
        let (dc, _) = new_ori.offset();
        let kick_col = self.col as i32 - dc;
        if self.can_occupy(kick_col, self.row as i32, new_ori, board) {
            self.col = kick_col as usize;
            self.orientation = new_ori;
            return true;
        }
        false
    }

    /// Check if a piece can occupy the given position.
    fn can_occupy(&self, col: i32, row: i32, ori: Orientation, board: &Board) -> bool {
        let cols = board.config.cols;
        let rows = board.config.rows;
        let (dc, dr) = ori.offset();
        let sc = col + dc;
        let sr = row + dr;

        // Bounds check
        if !(0..cols as i32).contains(&col) || !(0..cols as i32).contains(&sc) {
            return false;
        }
        if row < 0 || sr < 0 {
            return false;
        }

        let col_u = col as usize;
        let sc_u = sc as usize;
        let row_u = row as usize;
        let sr_u = sr as usize;

        // Height-based collision check
        if row_u < board.column_height(col_u) {
            return false;
        }
        if sr_u < board.column_height(sc_u) {
            return false;
        }

        // Cell-level collision check (top hidden row (rows-1) isolated puyo protection)
        if row_u < rows && board.get(col_u, row_u).is_occupied() {
            return false;
        }
        if sr_u < rows && board.get(sc_u, sr_u).is_occupied() {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GameConfig {
        GameConfig::default()
    }

    #[test]
    fn test_orientation_rotation() {
        assert_eq!(Orientation::North.rotate_cw(), Orientation::East);
        assert_eq!(Orientation::East.rotate_cw(), Orientation::South);
        assert_eq!(Orientation::South.rotate_cw(), Orientation::West);
        assert_eq!(Orientation::West.rotate_cw(), Orientation::North);

        assert_eq!(Orientation::North.rotate_ccw(), Orientation::West);
        assert_eq!(Orientation::West.rotate_ccw(), Orientation::South);
    }

    #[test]
    fn test_placement_satellite_col() {
        let c = cfg();
        let cols = c.cols;

        let p = Placement::new(1, Orientation::East);
        assert_eq!(p.satellite_col(cols), Some(2));

        let p = Placement::new(0, Orientation::West);
        assert_eq!(p.satellite_col(cols), None); // out of bounds

        let p = Placement::new(2, Orientation::East);
        assert_eq!(p.satellite_col(cols), None); // out of bounds

        let p = Placement::new(1, Orientation::North);
        assert_eq!(p.satellite_col(cols), Some(1)); // same column
    }

    #[test]
    fn test_falling_piece_spawn() {
        let c = cfg();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let fp = FallingPiece::spawn(piece, &c);
        assert_eq!(fp.col, c.spawn_col());
        assert_eq!(fp.orientation, Orientation::North);
    }

    fn board_with_heights(heights: &[usize]) -> Board {
        let mut board = Board::default();
        let cols = board.config.cols;
        for (col, &h) in heights.iter().enumerate().take(cols) {
            for row in 0..h {
                board.set(col, row, PuyoColor::Red);
            }
        }
        board
    }

    #[test]
    fn test_move_left_right() {
        let c = cfg();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece::spawn(piece, &c);
        let board = Board::default();

        assert!(fp.try_move_left(&board));
        assert_eq!(fp.col, 0);
        assert!(!fp.try_move_left(&board));
        assert_eq!(fp.col, 0);

        fp.col = 1;
        fp.orientation = Orientation::North;
        assert!(fp.try_move_right(&board));
        assert_eq!(fp.col, 2);
        assert!(!fp.try_move_right(&board));
    }

    #[test]
    fn test_rotate_wall_kick() {
        let c = cfg();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece::spawn(piece, &c);
        let board = Board::default();

        fp.col = 0;
        fp.orientation = Orientation::North;
        assert!(fp.try_rotate_cw(&board));
        assert_eq!(fp.orientation, Orientation::East);

        assert!(fp.try_rotate_cw(&board));
        assert_eq!(fp.orientation, Orientation::South);

        assert!(fp.try_rotate_cw(&board));
        assert_eq!(fp.orientation, Orientation::West);
    }

    #[test]
    fn test_move_blocked_by_existing_puyos() {
        let c = cfg();
        let cols = c.cols;
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 1,
            row: 2.0,
            orientation: Orientation::North,
        };

        let mut heights = vec![0; cols];
        heights[0] = 5;
        let board = board_with_heights(&heights);

        assert!(!fp.try_move_left(&board));
        assert_eq!(fp.col, 1);

        heights[2] = 5;
        let board = board_with_heights(&heights);
        assert!(!fp.try_move_right(&board));
        assert_eq!(fp.col, 1);

        fp.row = 6.0;
        assert!(fp.try_move_left(&board));
        assert_eq!(fp.col, 0);
    }

    #[test]
    fn test_move_blocked_by_satellite_collision() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 1,
            row: 2.0,
            orientation: Orientation::East,
        };

        let board = Board::default();
        assert!(fp.try_move_left(&board));
        assert_eq!(fp.col, 0);

        fp.col = 1;
        assert!(!fp.try_move_right(&board));
        assert_eq!(fp.col, 1);
    }

    #[test]
    fn test_rotate_blocked_by_existing_puyos() {
        let c = cfg();
        let cols = c.cols;
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 1,
            row: 2.0,
            orientation: Orientation::North,
        };

        let mut heights = vec![0; cols];
        heights[2] = 5;
        heights[0] = 5;
        let board = board_with_heights(&heights);

        assert!(!fp.try_rotate_cw(&board));
        assert_eq!(fp.orientation, Orientation::North);
    }

    #[test]
    fn test_move_blocked_by_isolated_top_hidden_row() {
        let c = cfg();
        let rows = c.rows;
        let visible_rows = c.visible_rows();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut board = Board::default();
        for row in 0..visible_rows {
            board.set(1, row, PuyoColor::Red);
        }
        board.set(1, rows - 1, PuyoColor::Green);

        let mut fp = FallingPiece {
            piece,
            col: 2,
            row: (rows - 2) as f32,
            orientation: Orientation::North,
        };
        assert!(!fp.try_move_left(&board));
        assert_eq!(fp.col, 2);
    }
}
