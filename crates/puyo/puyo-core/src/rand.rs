use crate::board::PuyoColor;
use crate::piece::Piece;

/// splitmix64 finalizer — mixes a u64 seed into a well-distributed hash.
fn splitmix64(s: u64) -> u64 {
    let s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
    s ^ (s >> 31)
}

/// 現在時刻からランダムなシードを生成する。
/// 呼び出しごとにカウンターを加算し、同一時刻でも異なる値を返す。
pub fn time_seed() -> u64 {
    use core::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    #[cfg(not(target_arch = "wasm32"))]
    let raw = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    #[cfg(target_arch = "wasm32")]
    let raw = (js_sys::Date::now() * 1_000_000.0) as u64;

    splitmix64(raw.wrapping_add(count))
}

/// 指定された seed と piece index から、決定論的にぷよを生成する。
/// 同じ seed / index / num_colors なら必ず同じ Piece を返す。
pub fn seeded_piece(seed: u64, index: u64, num_colors: usize) -> Piece {
    assert!(num_colors > 0);

    const GOLDEN_GAMMA: u64 = 0x9E3779B97F4A7C15;
    const SAT_SALT: u64 = 0xD1B54A32D192ED03;

    let base = seed.wrapping_add(index.wrapping_mul(GOLDEN_GAMMA));

    let axis_hash = splitmix64(base);
    let sat_hash = splitmix64(base ^ SAT_SALT);

    let axis = ((axis_hash % num_colors as u64) as u8) + 1;
    let sat = ((sat_hash % num_colors as u64) as u8) + 1;

    Piece::new(PuyoColor::from_u8(axis), PuyoColor::from_u8(sat))
}

/// time_seed() を使ってランダムなピースを生成する。
pub fn random_piece(num_colors: usize) -> Piece {
    let mut x = time_seed();
    let axis = ((x % num_colors as u64) as u8) + 1;
    x = (x ^ (x >> 30)).wrapping_mul(0x517cc1b727220a95);
    x = x ^ (x >> 27);
    let sat = ((x % num_colors as u64) as u8) + 1;
    Piece::new(PuyoColor::from_u8(axis), PuyoColor::from_u8(sat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor;
    use crate::config::GameConfig;

    #[test]
    fn test_random_piece_valid() {
        let num_colors = GameConfig::default().num_colors;
        for _ in 0..100 {
            let p = random_piece(num_colors);
            assert_ne!(p.axis_color, PuyoColor::Empty);
            assert_ne!(p.satellite_color, PuyoColor::Empty);
        }
    }

    #[test]
    fn test_time_seed_nonzero() {
        let s = time_seed();
        assert_ne!(s, 0);
    }
}
