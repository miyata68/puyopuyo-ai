use crate::board::{Group, MIN_GROUP_SIZE};

/// Chain power table (standard Puyo Puyo rules).
/// Index 0 = chain 1, index 1 = chain 2, etc.
const CHAIN_POWER: [u32; 24] = [
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544,
    576, 608, 640, 672,
];

/// Color bonus table based on number of different colors cleared simultaneously.
const COLOR_BONUS: [u32; 5] = [0, 0, 3, 6, 12];

/// Group bonus table based on number of puyos in a single connected group.
/// Index 0 = 4 puyos, index 1 = 5, etc.
const GROUP_BONUS: [u32; 8] = [0, 2, 3, 4, 5, 6, 7, 10];

/// Calculate score for a single chain step.
pub fn calculate_step_score(chain_num: u32, groups: &[Group]) -> u32 {
    if groups.is_empty() {
        return 0;
    }

    // Total number of puyos cleared
    let total_cleared: u32 = groups
        .iter()
        .filter(|g| g.color.is_normal_color())
        .map(|g| g.cells.len() as u32)
        .sum();

    // Chain power
    let cp_idx = (chain_num as usize)
        .saturating_sub(1)
        .min(CHAIN_POWER.len() - 1);
    let chain_power = CHAIN_POWER[cp_idx];

    // Color bonus: count distinct colors
    let mut colors_seen = [false; 5]; // index by PuyoColor as u8 (max Yellow=4)
    for group in groups {
        if group.color.is_normal_color() {
            colors_seen[group.color as usize] = true;
        }
    }
    let num_colors = colors_seen.iter().filter(|&&c| c).count();
    let color_bonus = COLOR_BONUS[num_colors.min(COLOR_BONUS.len() - 1)];

    // Group bonus: sum over each group
    let group_bonus: u32 = groups
        .iter()
        .filter(|g| g.color.is_normal_color())
        .map(|g| {
            let excess = g.cells.len().saturating_sub(MIN_GROUP_SIZE);
            GROUP_BONUS[excess.min(GROUP_BONUS.len() - 1)]
        })
        .sum();

    // Total bonus (minimum 1)
    let bonus = (chain_power + color_bonus + group_bonus).max(1);

    total_cleared * 10 * bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor;

    fn make_group(color: PuyoColor, count: usize) -> Group {
        let cells: Vec<(usize, usize)> = (0..count).map(|i| (0, i)).collect();
        Group { color, cells }
    }

    #[test]
    fn test_single_group_4_chain1() {
        // 4 puyos, chain 1: bonus = max(0+0+0, 1) = 1, score = 4*10*1 = 40
        let groups = vec![make_group(PuyoColor::Red, 4)];
        let score = calculate_step_score(1, &groups);
        assert_eq!(score, 40);
    }

    #[test]
    fn test_single_group_5_chain1() {
        // 5 puyos, chain 1: bonus = max(0+0+2, 1) = 2, score = 5*10*2 = 100
        let groups = vec![make_group(PuyoColor::Red, 5)];
        let score = calculate_step_score(1, &groups);
        assert_eq!(score, 100);
    }

    #[test]
    fn test_chain2_score() {
        // 4 puyos, chain 2: bonus = max(8+0+0, 1) = 8, score = 4*10*8 = 320
        let groups = vec![make_group(PuyoColor::Red, 4)];
        let score = calculate_step_score(2, &groups);
        assert_eq!(score, 320);
    }

    #[test]
    fn test_two_colors_bonus() {
        // Two groups of different colors, chain 1
        let groups = vec![
            make_group(PuyoColor::Red, 4),
            make_group(PuyoColor::Blue, 4),
        ];
        // bonus = max(0+3+0, 1) = 3, cleared = 8, score = 8*10*3 = 240
        let score = calculate_step_score(1, &groups);
        assert_eq!(score, 240);
    }

    #[test]
    fn test_empty_groups() {
        let score = calculate_step_score(1, &[]);
        assert_eq!(score, 0);
    }
}
