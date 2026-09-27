//! Fixed-size PvP observation. No absolute player ID, tick, or score is encoded.
use crate::placement::placement_to_index;
use crate::pvp::{MatchState, PlayerId, PlayerPhase};
pub const BOARD_CHANNELS: usize = 10;
pub const ROWS: usize = 14;
pub const COLS: usize = 6;
pub const BOARD_SIZE: usize = BOARD_CHANNELS * ROWS * COLS;
pub const CONTEXT_SIZE: usize = 68;
pub const NUM_ACTIONS: usize = 24;
pub fn opponent(id: PlayerId) -> PlayerId {
    match id {
        PlayerId::Player1 => PlayerId::Player2,
        PlayerId::Player2 => PlayerId::Player1,
    }
}
pub fn pvp_board_to_tensor_data(state: &MatchState, perspective: PlayerId) -> Vec<f32> {
    let mut data = vec![0.0; BOARD_SIZE];
    for (side, id) in [perspective, opponent(perspective)].into_iter().enumerate() {
        let board = &state.players[id.index()].board;
        for row in 0..ROWS {
            for col in 0..COLS {
                let color = board.get(col, row) as usize;
                if color > 0 {
                    data[(side * 5 + color - 1) * ROWS * COLS + row * COLS + col] = 1.0;
                }
            }
        }
    }
    data
}
/// 0..24 self queue, 24..48 opponent queue; 48..52 incoming;
/// 52..54 remainder, 54..56 all-clear, 56..64 phases, 64..66 chain, 66..68 drop_due.
pub fn pvp_context_to_tensor_data(state: &MatchState, perspective: PlayerId) -> Vec<f32> {
    let mut data = vec![0.0; CONTEXT_SIZE];
    for (side, id) in [perspective, opponent(perspective)].into_iter().enumerate() {
        let p = &state.players[id.index()];
        for next in 0..3 {
            let piece = state.piece_at(p.piece_index.wrapping_add(next as u64));
            for (half, color) in [piece.axis_color, piece.satellite_color]
                .into_iter()
                .enumerate()
            {
                data[side * 24 + next * 8 + half * 4 + color as usize - 1] = 1.0;
            }
        }
        data[48 + side * 2] = p.pending_garbage.min(300) as f32 / 30.0;
        data[49 + side * 2] = p.confirmed_garbage.min(300) as f32 / 30.0;
        data[52 + side] = p.attack_remainder as f32 / state.rules().garbage_rate as f32;
        data[54 + side] = u8::from(p.all_clear_bonus) as f32;
        let phase = match p.phase {
            PlayerPhase::Ready if p.garbage_drop_due && p.confirmed_garbage > 0 => 2,
            PlayerPhase::Ready => 0,
            PlayerPhase::Chaining => 1,
            PlayerPhase::Dead => 3,
        };
        data[56 + side * 4 + phase] = 1.0;
        data[64 + side] = p.chain_count.min(20) as f32 / 20.0;
        data[66 + side] = u8::from(p.garbage_drop_due) as f32;
    }
    data
}
pub fn pvp_valid_action_mask(state: &MatchState, perspective: PlayerId) -> [bool; NUM_ACTIONS] {
    let mut mask = [false; NUM_ACTIONS];
    for action in state.legal_actions(perspective) {
        mask[placement_to_index(&action)] = true;
    }
    mask
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor as C;
    #[test]
    fn channels_context_and_perspective() {
        let mut s = MatchState::new(23);
        s.players[0].board.set(1, 2, C::Red);
        s.players[0].board.set(2, 3, C::Garbage);
        s.players[1].board.set(3, 4, C::Red);
        s.players[1].board.set(4, 5, C::Garbage);
        s.players[1].piece_index = 7;
        s.players[0].pending_garbage = 60;
        s.players[0].confirmed_garbage = 30;
        s.players[0].attack_remainder = 35;
        s.players[0].all_clear_bonus = true;
        s.players[1].phase = PlayerPhase::Chaining;
        s.players[1].chain_count = 5;
        let b = pvp_board_to_tensor_data(&s, PlayerId::Player1);
        assert_eq!(b.len(), 840);
        assert_eq!(b.iter().sum::<f32>(), 4.0);
        for i in [
            2 * 6 + 1,
            4 * 84 + 3 * 6 + 2,
            5 * 84 + 4 * 6 + 3,
            9 * 84 + 5 * 6 + 4,
        ] {
            assert_eq!(b[i], 1.0);
        }
        let c = pvp_context_to_tensor_data(&s, PlayerId::Player1);
        assert_eq!(c.len(), 68);
        assert_eq!(&c[48..56], &[2., 1., 0., 0., 0.5, 0., 1., 0.]);
        assert_eq!(&c[56..64], &[0., 0., 1., 0., 0., 1., 0., 0.]);
        assert_eq!(c[65], 0.25);
        for (side, index) in [0u64, 7].into_iter().enumerate() {
            for n in 0..3 {
                let piece = s.piece_at(index + n);
                assert_eq!(
                    c[side * 24 + n as usize * 8 + piece.axis_color as usize - 1],
                    1.
                );
                assert_eq!(
                    c[side * 24 + n as usize * 8 + 4 + piece.satellite_color as usize - 1],
                    1.
                );
            }
        }
        let mut swapped = s.clone();
        swapped.players.swap(0, 1);
        assert_eq!(b, pvp_board_to_tensor_data(&swapped, PlayerId::Player2));
        assert_eq!(c, pvp_context_to_tensor_data(&swapped, PlayerId::Player2));
        s.tick = 999;
        s.players[0].score = 123456;
        s.players[0].max_chain = 19;
        assert_eq!(c, pvp_context_to_tensor_data(&s, PlayerId::Player1));
        assert_eq!(b, pvp_board_to_tensor_data(&s, PlayerId::Player1));
    }
    #[test]
    fn symmetric_input_has_no_id_and_masks_forced_actions() {
        let mut s = MatchState::new(1);
        assert_eq!(
            pvp_context_to_tensor_data(&s, PlayerId::Player1),
            pvp_context_to_tensor_data(&s, PlayerId::Player2)
        );
        s.players[0].phase = PlayerPhase::Dead;
        assert_eq!(pvp_context_to_tensor_data(&s, PlayerId::Player1)[59], 1.);
        assert!(!pvp_valid_action_mask(&s, PlayerId::Player1)
            .iter()
            .any(|&x| x));
        s.players[0].phase = PlayerPhase::Ready;
        s.players[0].confirmed_garbage = 1;
        assert!(!pvp_valid_action_mask(&s, PlayerId::Player1)
            .iter()
            .any(|&x| x));
    }
}
