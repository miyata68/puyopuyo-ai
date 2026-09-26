use std::collections::VecDeque;

use crate::config::GameConfig;

/// Puyo colors. Empty = no puyo in that cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PuyoColor {
    Empty = 0,
    Red = 1,
    Green = 2,
    Blue = 3,
    Yellow = 4,
    Garbage = 5,
}

/// All color variants in order (excluding Empty). NUM_COLORS selects the active subset.
const ALL_COLOR_VARIANTS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];

impl PuyoColor {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => PuyoColor::Red,
            2 => PuyoColor::Green,
            3 => PuyoColor::Blue,
            4 => PuyoColor::Yellow,
            5 => PuyoColor::Garbage,
            _ => PuyoColor::Empty,
        }
    }

    pub fn is_color(self) -> bool {
        self.is_normal_color()
    }

    pub fn is_normal_color(self) -> bool {
        matches!(self, Self::Red | Self::Green | Self::Blue | Self::Yellow)
    }

    pub fn is_garbage(self) -> bool {
        self == Self::Garbage
    }

    pub fn is_occupied(self) -> bool {
        self != Self::Empty
    }

    /// Returns the active color variants based on num_colors.
    pub fn active_colors(num_colors: usize) -> &'static [PuyoColor] {
        &ALL_COLOR_VARIANTS[..num_colors]
    }
}

pub use crate::config::MIN_GROUP_SIZE;

// ---- Chain types ----

/// A connected group of same-color puyos.
#[derive(Debug, Clone)]
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>, // (col, row)
}

/// Result of resolving all chains on a board.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub chain_count: u32,
    pub score: u32,
}

// ---- Board ----

/// Board stored in column-major flat Vec: cells[col * rows + row].
/// Row 0 is the bottom, the top hidden row (rows-1) is the top (hidden).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub config: GameConfig,
    cells: Vec<PuyoColor>,
}

impl Board {
    pub fn new(config: &GameConfig) -> Self {
        Board {
            cells: vec![PuyoColor::Empty; config.cols * config.rows],
            config: *config,
        }
    }

    fn idx(&self, col: usize, row: usize) -> usize {
        col * self.config.rows + row
    }

    /// Returns the height of a column (number of contiguous non-empty cells from bottom).
    pub fn column_height(&self, col: usize) -> usize {
        self.column_info(col).0
    }

    /// 列の高さと top hidden row 孤立ぷよの有無を同時に返す。
    pub fn column_info(&self, col: usize) -> (usize, bool) {
        let rows = self.config.rows;
        let mut height = rows;
        for row in 0..rows {
            if !self.get(col, row).is_occupied() {
                height = row;
                break;
            }
        }
        let isolated =
            self.get(col, rows - 1).is_occupied() && !self.get(col, rows - 2).is_occupied();
        (height, isolated)
    }

    /// Get the color at (col, row).
    pub fn get(&self, col: usize, row: usize) -> PuyoColor {
        self.cells[self.idx(col, row)]
    }

    /// Set the color at (col, row).
    pub fn set(&mut self, col: usize, row: usize, color: PuyoColor) {
        let idx = self.idx(col, row);
        self.cells[idx] = color;
    }

    /// Drop a puyo into a column. Returns the row it landed on.
    pub fn drop_puyo(&mut self, col: usize, color: PuyoColor) -> usize {
        let h = self.column_height(col);
        assert!(
            h < self.config.rows,
            "drop_puyo: column {col} is full (height={h})"
        );
        self.set(col, h, color);
        h
    }

    /// Apply gravity: make all puyos fall down to fill gaps.
    /// The top hidden row (rows-1) is excluded — puyos there stay until game over.
    pub fn apply_gravity(&mut self) {
        let cols = self.config.cols;
        let rows = self.config.rows;
        for col in 0..cols {
            let base = col * rows;
            let mut write = 0;
            for read in 0..(rows - 1) {
                let color = self.cells[base + read];
                if color.is_occupied() {
                    self.cells[base + write] = color;
                    if write != read {
                        self.cells[base + read] = PuyoColor::Empty;
                    }
                    write += 1;
                }
            }
        }
    }

    /// Check if game is over (spawn column has puyo above visible area).
    pub fn is_game_over(&self) -> bool {
        self.column_height(self.config.spawn_col()) >= self.config.visible_rows()
    }

    /// Flatten board to a Vec<u8> for WASM transfer. Column-major, bottom to top.
    pub fn to_flat(&self) -> Vec<u8> {
        self.cells.iter().map(|&c| c as u8).collect()
    }

    // ---- Chain detection & resolution ----

    /// Find all connected same-color groups on the visible board.
    pub fn find_connected_groups(&self) -> Vec<Group> {
        let cols = self.config.cols;
        let rows = self.config.rows;
        let visible_rows = self.config.visible_rows();
        let mut visited = vec![false; cols * rows];
        let mut groups = Vec::new();

        for col in 0..cols {
            for row in 0..visible_rows {
                let vi = self.idx(col, row);
                if !self.cells[vi].is_color() || visited[vi] {
                    continue;
                }
                let color = self.cells[vi];

                // BFS flood fill
                let mut queue = VecDeque::new();
                let mut cells = Vec::new();
                queue.push_back((col, row));
                visited[vi] = true;

                while let Some((c, r)) = queue.pop_front() {
                    cells.push((c, r));
                    for (dc, dr) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nc = c as i32 + dc;
                        let nr = r as i32 + dr;
                        if nc < 0 || nc >= cols as i32 || nr < 0 || nr >= visible_rows as i32 {
                            continue;
                        }
                        let (nc, nr) = (nc as usize, nr as usize);
                        let nvi = self.idx(nc, nr);
                        if !visited[nvi] && self.cells[nvi] == color {
                            visited[nvi] = true;
                            queue.push_back((nc, nr));
                        }
                    }
                }

                groups.push(Group { color, cells });
            }
        }

        groups
    }

    /// Find groups of min_group_size or larger (clearable groups).
    pub fn find_clearable_groups(&self) -> Vec<Group> {
        let min_group_size = self.config.min_group_size();
        self.find_connected_groups()
            .into_iter()
            .filter(|g| g.cells.len() >= min_group_size)
            .collect()
    }

    /// Resolve one chain step. Modifies board in-place.
    pub fn resolve_one_step(&mut self, chain_num: u32) -> Option<u32> {
        let groups = self.find_clearable_groups();
        if groups.is_empty() {
            return None;
        }

        for group in &groups {
            for &(col, row) in &group.cells {
                for (dc, dr) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (c, r) = (col as isize + dc, row as isize + dr);
                    if c >= 0
                        && r >= 0
                        && (c as usize) < self.config.cols
                        && (r as usize) < self.config.rows
                        && self.get(c as usize, r as usize).is_garbage()
                    {
                        self.set(c as usize, r as usize, PuyoColor::Empty);
                    }
                }
                self.set(col, row, PuyoColor::Empty);
            }
        }

        let step_score = crate::score::calculate_step_score(chain_num, &groups);
        self.apply_gravity();

        Some(step_score)
    }

    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(|c| !c.is_occupied())
    }

    /// Resolve all chains on the board. Modifies board in-place.
    pub fn resolve_chains(&mut self) -> ChainResult {
        let mut chain_count = 0u32;
        let mut score = 0u32;
        for chain_num in 1.. {
            match self.resolve_one_step(chain_num) {
                Some(step_score) => {
                    chain_count += 1;
                    score += step_score;
                }
                None => break,
            }
        }
        ChainResult { chain_count, score }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new(&GameConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GameConfig {
        GameConfig::default()
    }

    #[test]
    fn test_new_board_is_empty() {
        let c = cfg();
        let board = Board::new(&c);
        for col in 0..c.cols {
            assert_eq!(board.column_height(col), 0);
        }
    }

    #[test]
    fn test_drop_puyo() {
        let mut board = Board::new(&cfg());
        let row = board.drop_puyo(0, PuyoColor::Red);
        assert_eq!(row, 0);
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);

        let row = board.drop_puyo(0, PuyoColor::Blue);
        assert_eq!(row, 1);
        assert_eq!(board.get(0, 1), PuyoColor::Blue);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    #[should_panic(expected = "column 0 is full")]
    fn test_column_full() {
        let c = cfg();
        let mut board = Board::new(&c);
        for _ in 0..c.rows {
            board.drop_puyo(0, PuyoColor::Red);
        }
        board.drop_puyo(0, PuyoColor::Red);
    }

    #[test]
    fn test_apply_gravity() {
        let mut board = Board::new(&cfg());
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 2, PuyoColor::Blue);
        board.set(0, 5, PuyoColor::Green);
        board.apply_gravity();
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.get(0, 1), PuyoColor::Blue);
        assert_eq!(board.get(0, 2), PuyoColor::Green);
        assert_eq!(board.get(0, 3), PuyoColor::Empty);
    }

    #[test]
    fn test_game_over() {
        let c = cfg();
        let mut board = Board::new(&c);
        assert!(!board.is_game_over());
        for _ in 0..c.visible_rows() - 1 {
            board.drop_puyo(c.spawn_col(), PuyoColor::Red);
        }
        assert!(!board.is_game_over());
        board.drop_puyo(c.spawn_col(), PuyoColor::Red);
        assert!(board.is_game_over());
    }

    #[test]
    fn test_top_hidden_row_not_affected_by_gravity() {
        let c = cfg();
        let mut board = Board::new(&c);
        board.set(0, c.rows - 1, PuyoColor::Red);
        board.apply_gravity();
        assert_eq!(board.get(0, c.rows - 1), PuyoColor::Red);
        assert_eq!(board.get(0, 0), PuyoColor::Empty);
    }

    #[test]
    fn test_column_height_with_isolated_top_hidden_row() {
        let c = cfg();
        let mut board = Board::new(&c);
        board.set(0, c.rows - 1, PuyoColor::Red);
        assert_eq!(board.column_height(0), 0);
    }

    #[test]
    fn test_column_height_with_stack_and_top_hidden_row() {
        let c = cfg();
        let mut board = Board::new(&c);
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Blue);
        board.set(0, 2, PuyoColor::Green);
        board.set(0, c.rows - 1, PuyoColor::Blue);
        assert_eq!(board.column_height(0), 3);
    }

    #[test]
    fn test_drop_puyo_with_isolated_top_hidden_row() {
        let c = cfg();
        let mut board = Board::new(&c);
        board.set(0, c.rows - 1, PuyoColor::Red);
        let row = board.drop_puyo(0, PuyoColor::Blue);
        assert_eq!(row, 0);
        assert_eq!(board.get(0, 0), PuyoColor::Blue);
    }

    #[test]
    fn test_column_info_isolated() {
        let c = cfg();
        let mut board = Board::new(&c);
        assert_eq!(board.column_info(0), (0, false));
        board.set(0, c.rows - 1, PuyoColor::Red);
        assert_eq!(board.column_info(0), (0, true));
        board.set(0, c.rows - 2, PuyoColor::Blue);
        assert_eq!(board.column_info(0), (0, false));
        assert_eq!(board.column_info(1), (0, false));
    }

    #[test]
    fn test_to_flat() {
        let c = cfg();
        let board = Board::new(&c);
        let flat = board.to_flat();
        assert_eq!(flat.len(), c.cols * c.rows);
        assert!(flat.iter().all(|&v| v == 0));
    }

    // ---- Chain tests ----

    #[test]
    fn test_no_chain() {
        let mut board = Board::new(&cfg());
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Blue);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 0);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_single_group_clear() {
        let mut board = Board::new(&cfg());
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
        assert!(result.score > 0);
        assert_eq!(board.column_height(0), 0);
    }

    #[test]
    fn test_l_shape_group_clear() {
        let mut board = Board::new(&cfg());
        board.set(0, 0, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(2, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
    }

    #[test]
    fn test_two_chain() {
        let mut board = Board::new(&cfg());
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        board.drop_puyo(1, PuyoColor::Blue);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 2);
        assert_eq!(board.column_height(0), 0);
        assert_eq!(board.column_height(1), 0);
    }

    #[test]
    fn test_gravity_after_clear() {
        let mut board = Board::new(&cfg());
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        board.drop_puyo(0, PuyoColor::Green);
        board.resolve_chains();
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_l_shape_group() {
        let mut board = Board::new(&cfg());
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(2, 0, PuyoColor::Red);
        let groups = board.find_clearable_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].cells.len(), 4);
    }

    #[test]
    fn test_hidden_row_puyo_not_cleared() {
        let c = cfg();
        let mut board = Board::new(&c);
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        board.set(0, c.visible_rows(), PuyoColor::Red);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_hidden_row_puyo_falls_after_clear() {
        let c = cfg();
        let mut board = Board::new(&c);
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        board.set(0, 4, PuyoColor::Green);
        board.set(0, c.visible_rows(), PuyoColor::Green);
        board.resolve_chains();
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.get(0, 1), PuyoColor::Green);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    fn test_hidden_row_only_does_not_clear() {
        let c = cfg();
        let mut board = Board::new(&c);
        for col in 0..c.cols {
            board.set(col, c.visible_rows(), PuyoColor::Red);
        }
        board.set(0, c.visible_rows() + 1, PuyoColor::Red);
        let groups = board.find_clearable_groups();
        assert!(
            groups.is_empty(),
            "Hidden-row-only puyos should not form clearable groups"
        );
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 0);
    }
}
