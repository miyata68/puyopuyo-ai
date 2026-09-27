use serde::{Deserialize, Serialize};

// ─── Color Permutation Data Augmentation ───

/// 色の置換テーブル。perm[old_ch] = new_ch
pub type ColorPermutation = Vec<usize>;

/// num_colors! 通りの全色置換を返す（恒等置換を含む）。
pub fn all_color_permutations(num_colors: usize) -> Vec<ColorPermutation> {
    let mut result = Vec::new();
    let mut current = vec![0usize; num_colors];
    let mut used = vec![false; num_colors];
    generate_perms(&mut result, &mut current, &mut used, 0, num_colors);
    result
}

fn generate_perms(
    result: &mut Vec<ColorPermutation>,
    current: &mut Vec<usize>,
    used: &mut Vec<bool>,
    depth: usize,
    num_colors: usize,
) {
    if depth == num_colors {
        result.push(current.clone());
        return;
    }
    for i in 0..num_colors {
        if !used[i] {
            used[i] = true;
            current[depth] = i;
            generate_perms(result, current, used, depth + 1, num_colors);
            used[i] = false;
        }
    }
}

/// board_data にインプレースで色置換を適用する。
/// board_data は [num_colors ch][rows][cols] floats。色チャネルを入れ替える。
pub fn apply_color_perm_board(board_data: &mut [f32], perm: &[usize]) {
    let num_colors = perm.len();
    let plane_size = board_data.len() / num_colors;

    let mut color_planes = vec![0.0f32; num_colors * plane_size];
    color_planes.copy_from_slice(&board_data[..num_colors * plane_size]);

    for (old_ch, &new_ch) in perm.iter().enumerate().take(num_colors) {
        let src_start = old_ch * plane_size;
        let dst_start = new_ch * plane_size;
        board_data[dst_start..dst_start + plane_size]
            .copy_from_slice(&color_planes[src_start..src_start + plane_size]);
    }
}

/// context_data にインプレースで色置換を適用する。
/// context_data は 6つのnum_colors要素one-hotブロック = piece_tensor_size floats。
pub fn apply_color_perm_context(context_data: &mut [f32], perm: &[usize]) {
    let num_colors = perm.len();
    let piece_tensor_size = context_data.len();

    for block_start in (0..piece_tensor_size).step_by(num_colors) {
        let mut tmp = vec![0.0f32; num_colors];
        tmp.copy_from_slice(&context_data[block_start..block_start + num_colors]);
        for old_ch in 0..num_colors {
            context_data[block_start + perm[old_ch]] = tmp[old_ch];
        }
    }
}

/// A single training sample: board state + context info + chosen action + value estimate.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sample {
    /// Encoded board state (tensor_size floats).
    pub board_data: Vec<f32>,
    /// Encoded context (piece_tensor_size floats).
    pub context_data: Vec<f32>,
    /// Action index chosen by the teacher evaluator.
    pub action_index: u8,
    /// Value target: score estimate from the teacher evaluator.
    pub value_target: f32,
}

pub use az_framework::data::{AlphaZeroDataset, AlphaZeroSample};

/// Dataset of training samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: Dataset = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::config::GameConfig;

    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    fn identity_perm(num_colors: usize) -> ColorPermutation {
        (0..num_colors).collect()
    }

    #[test]
    fn test_all_permutations_count_and_unique() {
        let gc = GameConfig::default();
        let num_colors = gc.num_colors;
        let perms = all_color_permutations(num_colors);
        let expected_count = factorial(num_colors);
        assert_eq!(perms.len(), expected_count);
        for i in 0..perms.len() {
            let mut sorted = perms[i].clone();
            sorted.sort();
            assert_eq!(sorted, identity_perm(num_colors));
            for j in (i + 1)..perms.len() {
                assert_ne!(perms[i], perms[j]);
            }
        }
    }

    #[test]
    fn test_identity_permutation_exists() {
        let gc = GameConfig::default();
        let num_colors = gc.num_colors;
        let perms = all_color_permutations(num_colors);
        let id = identity_perm(num_colors);
        assert!(perms.iter().any(|p| *p == id));
    }

    #[test]
    fn test_board_perm_identity() {
        let gc = GameConfig::default();
        let tensor_size = gc.tensor_size();
        let num_colors = gc.num_colors;
        let mut data = vec![0.0f32; tensor_size];
        data[0] = 1.0;
        let original = data.clone();
        apply_color_perm_board(&mut data, &identity_perm(num_colors));
        assert_eq!(data, original);
    }

    #[test]
    fn test_board_perm_swap_first_two() {
        let gc = GameConfig::default();
        let tensor_size = gc.tensor_size();
        let num_colors = gc.num_colors;
        let plane_size = gc.rows * gc.cols;
        let mut data = vec![0.0f32; tensor_size];
        data[0] = 1.0; // ch0, position 0
        let mut perm = identity_perm(num_colors);
        perm.swap(0, 1);
        apply_color_perm_board(&mut data, &perm);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[plane_size], 1.0);
    }

    #[test]
    fn test_context_perm_swap() {
        let gc = GameConfig::default();
        let nc = gc.num_colors;
        let piece_tensor_size = gc.piece_tensor_size();
        let mut ctx = vec![0.0f32; piece_tensor_size];
        ctx[0] = 1.0;
        ctx[nc + 2 % nc] = 1.0;
        let mut perm = identity_perm(nc);
        perm.swap(0, 1);
        apply_color_perm_context(&mut ctx, &perm);
        assert_eq!(ctx[0], 0.0);
        assert_eq!(ctx[1], 1.0);
    }
}
