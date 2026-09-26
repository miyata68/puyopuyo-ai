//! ゲームパラメータの一元管理モジュール。
//!
//! `GameConfig` でボードサイズ・色数を実行時に指定可能。

use serde::{Deserialize, Serialize};

/// 連鎖で消えるために必要な同色ぷよの最小接続数。
pub const MIN_GROUP_SIZE: usize = 4;

/// ゲームの基本パラメータ。盤面サイズ・色数を実行時に指定できる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameConfig {
    /// ボードの列数（横幅）。
    pub cols: usize,
    /// ボードの行数（表示行 + 隠し2行）。
    pub rows: usize,
    /// 使用する色の数（1〜4）。PuyoColor の先頭 N 色を使う。
    pub num_colors: usize,
}

impl GameConfig {
    pub fn new(cols: usize, rows: usize, num_colors: usize) -> Self {
        assert!(cols >= 1, "cols must be >= 1");
        assert!(rows >= 3, "rows must be >= 3");
        assert!((1..=4).contains(&num_colors), "num_colors must be 1..=4");
        Self {
            cols,
            rows,
            num_colors,
        }
    }

    /// 表示行数。上2行は隠し行。
    pub fn visible_rows(&self) -> usize {
        self.rows - 2
    }

    /// ぷよの出現列（0-indexed、ボード中央）。
    pub fn spawn_col(&self) -> usize {
        (self.cols - 1) / 2
    }

    /// アクション空間のサイズ: cols × 4方向。
    pub fn num_actions(&self) -> usize {
        self.cols * 4
    }

    /// 入力チャンネル数: 色ごとの one-hot。
    pub fn num_channels(&self) -> usize {
        self.num_colors
    }

    /// 盤面テンソルのフラットサイズ。
    pub fn tensor_size(&self) -> usize {
        self.num_channels() * self.rows * self.cols
    }

    /// ピースエンコーディングのサイズ: 3ピース × 2色 × num_colors one-hot。
    pub fn piece_tensor_size(&self) -> usize {
        3 * 2 * self.num_colors
    }

    /// コンテキストテンソルサイズ。
    pub fn context_tensor_size(&self) -> usize {
        self.piece_tensor_size()
    }

    /// 連鎖で消えるために必要な同色ぷよの最小接続数。
    pub fn min_group_size(&self) -> usize {
        MIN_GROUP_SIZE
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            cols: 3,
            rows: 8,
            num_colors: 3,
        }
    }
}
