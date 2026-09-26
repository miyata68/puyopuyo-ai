use crate::board::{Board, ChainResult};
use crate::piece::{Orientation, Piece, Placement};

/// Place a piece directly onto a board (no GameState overhead).
pub(crate) fn place_piece_on_board(board: &mut Board, piece: &Piece, placement: &Placement) {
    let (dc, _dr) = placement.orientation.offset();

    match placement.orientation {
        Orientation::North => {
            board.drop_puyo(placement.col, piece.axis_color);
            let sat_col = (placement.col as i32 + dc) as usize;
            let sat_h = board.column_height(sat_col);
            if sat_h < board.config.rows && !board.get(sat_col, sat_h).is_occupied() {
                board.drop_puyo(sat_col, piece.satellite_color);
            }
        }
        Orientation::South => {
            let sat_col = (placement.col as i32 + dc) as usize;
            board.drop_puyo(sat_col, piece.satellite_color);
            board.drop_puyo(placement.col, piece.axis_color);
        }
        Orientation::East | Orientation::West => {
            board.drop_puyo(placement.col, piece.axis_color);
            let sat_col = (placement.col as i32 + dc) as usize;
            board.drop_puyo(sat_col, piece.satellite_color);
        }
    }
}

/// Simulate placing a piece on a board clone. Returns the resulting board and chain result.
pub fn simulate_placement(
    board: &Board,
    piece: &Piece,
    placement: &Placement,
) -> (Board, ChainResult) {
    let mut b = board.clone();
    place_piece_on_board(&mut b, piece, placement);
    let chain_result = b.resolve_chains();
    (b, chain_result)
}

/// Compute which columns are reachable from the spawn column.
/// A column with height >= ROWS - 1 blocks entry and further traversal.
fn compute_reachable_columns(board: &Board) -> Vec<bool> {
    let cols = board.config.cols;
    let rows = board.config.rows;
    let spawn_col = board.config.spawn_col();
    let mut reachable = vec![false; cols];

    if board.column_height(spawn_col) >= rows - 1 {
        return reachable;
    }

    let is_passable = |col: &usize| board.column_height(*col) < rows - 1;

    for col in std::iter::once(spawn_col)
        .chain((0..spawn_col).rev().take_while(is_passable))
        .chain((spawn_col + 1..cols).take_while(is_passable))
    {
        reachable[col] = true;
    }

    reachable
}

/// Enumerate all legal placements for a piece on the given board.
/// Maximum COLS*2 + (COLS-1)*2 placements: North/South × COLS + East/West × (COLS-1).
///
/// Placement rules:
/// 1. The axis puyo must not land at row ROWS-1 (the topmost hidden row).
/// 2. Each column involved must be reachable from the spawn column (SPAWN_COL).
///    A column with height >= ROWS-1 blocks traversal.
pub fn enumerate_placements(board: &Board, piece: &Piece) -> Vec<Placement> {
    enumerate_placements_impl(board, piece, false)
}

/// Tsu placements: double rotation permits South even between two blocked sides.
/// Reachability, landing height, and same-color deduplication remain unchanged.
pub fn enumerate_tsu_placements(board: &Board, piece: &Piece) -> Vec<Placement> {
    enumerate_placements_impl(board, piece, true)
}

fn enumerate_placements_impl(
    board: &Board,
    piece: &Piece,
    allow_double_rotation: bool,
) -> Vec<Placement> {
    let cols = board.config.cols;
    let rows = board.config.rows;
    let reachable = compute_reachable_columns(board);

    // North: axis on bottom, satellite above. h + 2 <= max_rows ensures both fit.
    // If the top hidden row (rows-1) has an isolated puyo, satellite cannot go there.
    let north = (0..cols)
        .filter(|&col| {
            let (h, isolated) = board.column_info(col);
            let max_rows = if isolated { rows - 1 } else { rows };
            reachable[col] && h + 2 <= max_rows
        })
        .map(|col| Placement::new(col, Orientation::North));

    // South: satellite on bottom, axis above. Axis at h+1 must be < rows-1.
    // Additionally, South requires rotating through East or West from spawn (North).
    // Tsu double rotation bypasses only the two-blocked-sides restriction.
    let south = (0..cols)
        .filter(|&col| {
            if !reachable[col] || board.column_height(col) + 2 > rows - 1 {
                return false;
            }
            let left_blocked = col == 0 || board.column_height(col - 1) >= rows - 1;
            let right_blocked = col == cols - 1 || board.column_height(col + 1) >= rows - 1;
            allow_double_rotation || !(left_blocked && right_blocked)
        })
        .map(|col| Placement::new(col, Orientation::South));

    // East: axis at col, satellite at col+1. Axis must be < rows-1.
    // If satellite column has an isolated puyo at the top hidden row, max height is reduced.
    let east = (0..cols - 1)
        .filter(|&col| {
            let (sat_h, sat_isolated) = board.column_info(col + 1);
            let sat_max = if sat_isolated { rows - 1 } else { rows };
            reachable[col]
                && reachable[col + 1]
                && board.column_height(col) < rows - 1
                && sat_h < sat_max
        })
        .map(|col| Placement::new(col, Orientation::East));

    // West: axis at col, satellite at col-1. Axis must be < rows-1.
    // If satellite column has an isolated puyo at the top hidden row, max height is reduced.
    let west = (1..cols)
        .filter(|&col| {
            let (sat_h, sat_isolated) = board.column_info(col - 1);
            let sat_max = if sat_isolated { rows - 1 } else { rows };
            reachable[col]
                && reachable[col - 1]
                && board.column_height(col) < rows - 1
                && sat_h < sat_max
        })
        .map(|col| Placement::new(col, Orientation::West));

    let placements: Vec<Placement> = north.chain(south).chain(east).chain(west).collect();

    // Deduplicate: if both colors are the same, North==South and East(col)==West(col+1).
    if piece.axis_color == piece.satellite_color {
        let mut seen = std::collections::HashSet::new();
        return placements
            .into_iter()
            .filter(|p| seen.insert(normalize_placement(p)))
            .collect();
    }

    placements
}

/// Convert a Placement to a flat index: col * 4 + orientation.as_u8().
pub fn placement_to_index(p: &Placement) -> usize {
    p.col * 4 + p.orientation.as_u8() as usize
}

/// Convert a flat index back to a Placement.
/// Panics if index >= num_actions.
pub fn index_to_placement(index: usize, cols: usize) -> Placement {
    let num_actions = cols * 4;
    assert!(index < num_actions, "index out of range: {}", index);
    let col = index / 4;
    let ori = match index % 4 {
        0 => Orientation::North,
        1 => Orientation::East,
        2 => Orientation::South,
        3 => Orientation::West,
        _ => unreachable!(),
    };
    Placement::new(col, ori)
}

/// Compute a valid-action mask from enumerate_placements.
/// Returns Vec<bool> of length num_actions where true = valid placement.
pub fn compute_valid_mask(board: &Board, piece: &Piece) -> Vec<bool> {
    let num_actions = board.config.num_actions();
    let mut mask = vec![false; num_actions];
    for p in enumerate_placements(board, piece) {
        mask[placement_to_index(&p)] = true;
    }
    mask
}

/// Normalize a placement for deduplication when both colors are the same.
/// Returns (min_col, max_col, is_vertical).
fn normalize_placement(p: &Placement) -> (usize, usize, bool) {
    let (dc, _dr) = p.orientation.offset();
    let sat_col = (p.col as i32 + dc) as usize;
    let is_vertical = matches!(p.orientation, Orientation::North | Orientation::South);

    if is_vertical {
        (p.col, p.col, true)
    } else {
        let min_c = p.col.min(sat_col);
        let max_c = p.col.max(sat_col);
        (min_c, max_c, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor;
    use crate::config::GameConfig;

    fn cfg() -> GameConfig {
        GameConfig::default()
    }

    #[test]
    fn test_empty_board_placements() {
        let c = cfg();
        let cols = c.cols;
        let board = Board::default();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);
        let expected = cols * 2 + (cols - 1) * 2;
        assert_eq!(placements.len(), expected);
    }

    #[test]
    fn test_same_color_dedup() {
        let c = cfg();
        let cols = c.cols;
        let board = Board::default();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let placements = enumerate_placements(&board, &piece);
        let expected = cols + (cols - 1);
        assert_eq!(placements.len(), expected);
    }

    #[test]
    fn test_full_column_reduces_placements() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let mut board = Board::default();
        for _ in 0..rows {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);
        let remaining = cols - 1;
        let expected = remaining * 2 + remaining.saturating_sub(1) + remaining.saturating_sub(1);
        assert_eq!(placements.len(), expected);
    }

    #[test]
    fn test_south_blocked_at_visible_height() {
        let c = cfg();
        let cols = c.cols;
        let visible_rows = c.visible_rows();
        let mut board = Board::default();
        for _ in 0..visible_rows {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::South));

        assert!(placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_at_height_below_visible() {
        let c = cfg();
        let cols = c.cols;
        let visible_rows = c.visible_rows();
        let mut board = Board::default();
        for _ in 0..visible_rows - 1 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_high_column_blocks_traversal_left() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements.iter().any(
            |p| p.col == 0 && matches!(p.orientation, Orientation::North | Orientation::South)
        ));

        assert!(placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_high_column_blocks_traversal_right() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements.iter().any(|p| p.col == cols - 1
            && matches!(p.orientation, Orientation::North | Orientation::South)));

        assert!(placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::North));
    }

    #[test]
    fn test_east_west_satellite_reachability() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::East));

        assert!(placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::West));

        assert!(placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::East));
    }

    #[test]
    fn test_north_blocked_by_isolated_top_row() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let mut board = Board::default();
        for _ in 0..rows - 2 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        board.set(cols - 1, rows - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_north_allowed_without_isolated_top_row() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let visible_rows = c.visible_rows();
        let mut board = Board::default();
        for _ in 0..visible_rows / 2 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        board.set(cols - 1, rows - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(placements
            .iter()
            .any(|p| p.col == cols - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_blocked_when_both_neighbors_high() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(0, PuyoColor::Red);
            board.drop_puyo(cols - 1, PuyoColor::Blue);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::South));

        assert!(placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_when_one_neighbor_high() {
        let c = cfg();
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::South));
    }

    #[test]
    fn test_south_blocked_at_boundary_col0() {
        let c = cfg();
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(spawn_col, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_east_west_blocked_by_isolated_top_row_satellite() {
        let c = cfg();
        let cols = c.cols;
        let rows = c.rows;
        let spawn_col = c.spawn_col();
        let mut board = Board::default();
        for _ in 0..rows - 1 {
            board.drop_puyo(cols - 1, PuyoColor::Red);
        }
        board.set(cols - 1, rows - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        assert!(!placements
            .iter()
            .any(|p| p.col == spawn_col && p.orientation == Orientation::East));
    }

    #[test]
    fn test_placement_to_index_roundtrip() {
        let c = cfg();
        let cols = c.cols;
        for col in 0..cols {
            for ori in [
                Orientation::North,
                Orientation::East,
                Orientation::South,
                Orientation::West,
            ] {
                let p = Placement::new(col, ori);
                let idx = placement_to_index(&p);
                let p2 = index_to_placement(idx, cols);
                assert_eq!(p, p2);
            }
        }
    }

    #[test]
    fn test_compute_valid_mask_empty_board() {
        let c = cfg();
        let cols = c.cols;
        let board = Board::default();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mask = compute_valid_mask(&board, &piece);
        let count = mask.iter().filter(|&&v| v).count();
        let expected = cols * 2 + (cols - 1) * 2;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_compute_valid_mask_same_color() {
        let c = cfg();
        let cols = c.cols;
        let board = Board::default();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let mask = compute_valid_mask(&board, &piece);
        let count = mask.iter().filter(|&&v| v).count();
        let expected = cols + (cols - 1);
        assert_eq!(count, expected);
    }
}
