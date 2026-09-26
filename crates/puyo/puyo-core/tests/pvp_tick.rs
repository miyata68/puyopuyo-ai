use puyo_core::board::{Board, Group, PuyoColor as C};
use puyo_core::config::GameConfig;
use puyo_core::piece::{Orientation, Piece, Placement};
use puyo_core::placement::{enumerate_placements, enumerate_tsu_placements};
use puyo_core::pvp::*;
use PlayerId::{Player1 as P1, Player2 as P2};
fn board() -> Board {
    Board::new(&GameConfig::new(6, 14, 4))
}
fn chain(n: usize) -> Board {
    let mut b = board();
    let colors = [C::Red, C::Green, C::Blue, C::Yellow, C::Red];
    for i in 0..n {
        for _ in 0..if i == 0 { 4 } else { 3 } {
            b.drop_puyo(i, colors[i]);
        }
        if i + 1 < n {
            b.drop_puyo(i, colors[i + 1]);
        }
    }
    let mut verify = b.clone();
    assert_eq!(verify.resolve_chains().chain_count, n as u32);
    b
}
fn set_chain(m: &mut MatchState, id: PlayerId, n: usize) {
    let p = &mut m.players[id.index()];
    p.board = chain(n);
    p.phase = PlayerPhase::Chaining;
}
fn action(m: &MatchState, id: PlayerId) -> Option<Placement> {
    m.legal_actions(id).last().copied()
}
fn advance(m: &mut MatchState) -> MatchStepResult {
    m.step(action(m, P1), action(m, P2)).unwrap()
}
#[test]
fn garbage_is_occupied_but_never_a_color_group() {
    let mut b = board();
    for c in 0..6 {
        b.drop_puyo(c, C::Garbage);
    }
    assert!(!C::Garbage.is_color());
    assert!(C::Garbage.is_occupied());
    assert!(C::Garbage.is_garbage());
    assert_eq!(C::from_u8(5), C::Garbage);
    assert_eq!(b.column_height(0), 1);
    assert_eq!(b.resolve_chains().score, 0);
    assert_eq!(b.column_height(0), 1);
    b.set(0, 3, C::Garbage);
    b.apply_gravity();
    assert_eq!(b.column_height(0), 2);
    assert!(puyo_core::state::board_to_tensor_data(&b)
        .iter()
        .all(|v| *v == 0.0));
}
#[test]
fn adjacent_garbage_clears_without_recursive_clear_or_score() {
    let mut b = board();
    for r in 0..4 {
        b.set(0, r, C::Red);
    }
    b.set(1, 0, C::Garbage);
    b.set(2, 0, C::Garbage);
    assert_eq!(b.resolve_one_step(1), Some(40));
    assert_eq!(b.get(1, 0), C::Empty);
    assert_eq!(b.get(2, 0), C::Garbage);
    let groups = [Group {
        color: C::Garbage,
        cells: vec![(0, 0); 6],
    }];
    assert_eq!(puyo_core::score::calculate_step_score(1, &groups), 0);
}
#[test]
fn all_clear_awarded_consumed_and_reearned() {
    let mut m = MatchState::new(10);
    set_chain(&mut m, P1, 1);
    assert_eq!(advance(&mut m).players[0].generated_attack, 0);
    assert!(m.players[0].all_clear_bonus);
    m.players[0].chain_count = 0;
    set_chain(&mut m, P1, 1);
    assert_eq!(advance(&mut m).players[0].generated_attack, 31);
    assert_eq!(m.players[0].attack_remainder, 10);
    assert!(m.players[0].all_clear_bonus);
    m.players[0].chain_count = 0;
    set_chain(&mut m, P1, 1);
    m.players[0].board.set(5, 0, C::Garbage);
    assert_eq!(advance(&mut m).players[0].generated_attack, 30);
    assert!(!m.players[0].all_clear_bonus);
}
#[test]
fn same_seed_and_index_shared_independent_of_progress() {
    let a = MatchState::new(100);
    let mut b = a.clone();
    b.players[1].piece_index = 400;
    for i in 0..1000 {
        assert_eq!(a.piece_at(i), b.piece_at(i));
    }
    assert_eq!(b.current_piece(P2), a.piece_at(400));
}
#[test]
fn five_chain_steps_allow_five_opponent_placements_and_delay_pending() {
    let mut m = MatchState::new(42);
    set_chain(&mut m, P1, 5);
    for n in 1..=5 {
        let e = advance(&mut m);
        assert_eq!(e.players[0].chain_step, n);
        assert!(e.players[1].placed);
        assert_eq!(e.players[1].dropped, 0);
        assert_eq!(m.players[1].piece_index, u64::from(n));
        assert_eq!(m.players[0].piece_index, 0);
        if n > 1 && n < 5 {
            assert!(m.players[1].pending_garbage > 0);
            assert_eq!(m.players[1].confirmed_garbage, 0);
        }
    }
    assert_eq!(m.players[0].phase, PlayerPhase::Ready);
    assert_eq!(m.players[1].pending_garbage, 0);
    assert!(m.players[1].confirmed_garbage > 0);
}
#[test]
fn five_vs_three_has_no_extra_chain_completion_tick() {
    let mut m = MatchState::new(42);
    set_chain(&mut m, P1, 5);
    set_chain(&mut m, P2, 3);
    for n in 1..=5 {
        let e = advance(&mut m);
        assert_eq!(e.players[0].chain_step, n);
        if n <= 3 {
            assert_eq!(e.players[1].chain_step, n);
        } else {
            assert!(e.players[1].placed);
        }
    }
    assert_eq!(m.players[1].piece_index, 2);
}
#[test]
fn own_chain_offsets_confirmed_then_pending() {
    let mut m = MatchState::new(1);
    set_chain(&mut m, P1, 1);
    m.players[0].all_clear_bonus = true;
    m.players[0].confirmed_garbage = 10;
    m.players[0].pending_garbage = 25;
    let e = advance(&mut m);
    assert_eq!(e.players[0].offset, 30);
    assert_eq!(e.players[0].sent, 0);
    assert_eq!(m.players[0].confirmed_garbage, 0);
    assert_eq!(m.players[0].pending_garbage, 5);
}
#[test]
fn swapping_players_and_actions_swaps_attack_results() {
    let mut a = MatchState::new(7);
    set_chain(&mut a, P1, 5);
    set_chain(&mut a, P2, 3);
    a.players[0].all_clear_bonus = true;
    a.players[1].pending_garbage = 4;
    let mut b = a.clone();
    b.players.swap(0, 1);
    for _ in 0..3 {
        let ea = advance(&mut a);
        let eb = advance(&mut b);
        assert_eq!(ea.players, [eb.players[1], eb.players[0]]);
        assert_eq!(a.players[0], b.players[1]);
        assert_eq!(a.players[1], b.players[0]);
    }
}
#[test]
fn drop_cap_and_piece_opportunity_between_drops() {
    let mut m = MatchState::new(8);
    m.players[0].confirmed_garbage = 60;
    let e = advance(&mut m);
    assert_eq!(e.players[0].dropped, 30);
    assert!(!e.players[0].placed);
    assert_eq!(m.players[0].confirmed_garbage, 30);
    assert!(m.requires_action(P1));
    assert!(advance(&mut m).players[0].placed);
    assert_eq!(advance(&mut m).players[0].dropped, 30);
}
#[test]
fn seventeen_drop_has_two_rows_and_five_unique_extra_columns() {
    for seed in 0..30 {
        let mut a = MatchState::new(seed);
        a.players[0].confirmed_garbage = 17;
        let mut b = a.clone();
        assert_eq!(advance(&mut a), advance(&mut b));
        assert_eq!(a, b);
        let heights: Vec<_> = (0..6)
            .map(|c| a.players[0].board.column_height(c))
            .collect();
        assert_eq!(heights.iter().filter(|&&h| h == 3).count(), 5);
        assert_eq!(heights.iter().filter(|&&h| h == 2).count(), 1);
    }
}
#[test]
fn confirmed_does_not_fall_during_chain() {
    let mut m = MatchState::new(2);
    set_chain(&mut m, P1, 3);
    m.players[0].confirmed_garbage = 100;
    for _ in 0..3 {
        assert_eq!(advance(&mut m).players[0].dropped, 0);
    }
    assert!(advance(&mut m).players[0].dropped > 0);
}
fn near_death(m: &mut MatchState, id: PlayerId) {
    for _ in 0..11 {
        m.players[id.index()].board.drop_puyo(2, C::Garbage);
    }
    m.players[id.index()].confirmed_garbage = 6;
}
#[test]
fn death_cell_after_drop_and_winner_value() {
    let mut m = MatchState::new(2);
    near_death(&mut m, P1);
    advance(&mut m);
    assert_eq!(m.result(), MatchResult::Player2Win);
    assert_eq!(m.winner(), Some(P2));
    assert_eq!(m.outcome_for(P1), Some(-1));
    assert_eq!(m.outcome_for(P2), Some(1));
    assert_eq!(m.step(None, None), Err(MatchError::Finished));
}
#[test]
fn simultaneous_death_is_draw() {
    let mut m = MatchState::new(2);
    near_death(&mut m, P1);
    near_death(&mut m, P2);
    advance(&mut m);
    assert_eq!(m.result(), MatchResult::Draw);
    assert_eq!(m.outcome_for(P1), Some(0));
}
#[test]
fn full_non_death_column_does_not_immediately_kill() {
    let mut m = MatchState::new(3);
    for r in 0..14 {
        m.players[0].board.set(0, r, C::Garbage);
    }
    m.players[0].confirmed_garbage = 6;
    let e = advance(&mut m);
    assert_eq!(m.result(), MatchResult::Ongoing);
    assert_eq!(m.players[0].phase, PlayerPhase::Ready);
    assert_eq!(e.players[0].dropped, 6);
    assert_eq!(m.players[0].confirmed_garbage, 0);
    assert_eq!(m.players[0].board.column_height(0), 14);
    let heights: Vec<_> = (1..6)
        .map(|c| m.players[0].board.column_height(c))
        .collect();
    assert_eq!(heights.iter().sum::<usize>(), 6);
    assert_eq!(heights.iter().filter(|&&h| h == 2).count(), 1);
    assert_eq!(heights.iter().filter(|&&h| h == 1).count(), 4);
}
#[test]
fn invalid_actions_are_atomic_and_explicit() {
    let mut m = MatchState::new(1);
    let before = m.clone();
    assert_eq!(
        m.step(action(&m, P1), None),
        Err(MatchError::MissingAction(P2))
    );
    assert_eq!(m, before);
    assert_eq!(
        m.step(
            action(&m, P1),
            Some(Placement::new(usize::MAX, Orientation::East))
        ),
        Err(MatchError::IllegalPlacement(P2))
    );
    assert_eq!(m, before);
    set_chain(&mut m, P1, 3);
    let before = m.clone();
    assert_eq!(
        m.step(Some(Placement::new(0, Orientation::North)), action(&m, P2)),
        Err(MatchError::UnexpectedAction(P1))
    );
    assert_eq!(m, before);
    m.players[0].phase = PlayerPhase::Ready;
    m.players[0].confirmed_garbage = 1;
    assert_eq!(
        m.step(Some(Placement::new(0, Orientation::North)), action(&m, P2)),
        Err(MatchError::UnexpectedAction(P1))
    );
}
#[test]
fn invalid_rules_are_rejected() {
    assert_eq!(
        MatchState::with_rules(
            1,
            TsuRules {
                garbage_rate: 0,
                ..Default::default()
            }
        ),
        Err(MatchError::InvalidRules)
    );
    assert_eq!(
        MatchState::with_rules(
            1,
            TsuRules {
                max_garbage_per_drop: 31,
                ..Default::default()
            }
        ),
        Err(MatchError::InvalidRules)
    );
}
#[test]
fn placement_starts_chain_without_clearing_until_next_tick() {
    let mut m = MatchState::new(5);
    let piece = m.current_piece(P1);
    for r in 0..3 {
        m.players[0].board.set(0, r, piece.axis_color);
    }
    let e = m
        .step(Some(Placement::new(0, Orientation::North)), action(&m, P2))
        .unwrap();
    assert!(e.players[0].placed);
    assert_eq!(e.players[0].score, 0);
    assert_eq!(m.players[0].phase, PlayerPhase::Chaining);
    assert_eq!(advance(&mut m).players[0].chain_step, 1);
}
#[test]
fn solo_full_resolution_matches_repeated_chain_steps() {
    let b = chain(5);
    let mut solo = b.clone();
    let expected = solo.resolve_chains();
    let mut stepped = b;
    let mut score = 0;
    for n in 1..=5 {
        score += stepped.resolve_one_step(n).unwrap();
    }
    assert_eq!(stepped.resolve_one_step(6), None);
    assert_eq!(stepped, solo);
    assert_eq!(score, expected.score);
}

#[test]
fn chain_can_rescue_occupied_death_cell_before_death_check() {
    let mut m = MatchState::new(17);
    for r in 0..11 {
        m.players[0].board.set(2, r, C::Garbage);
    }
    m.players[0].board.set(2, 11, C::Red);
    for r in 0..8 {
        m.players[0].board.set(1, r, C::Garbage);
    }
    for r in 8..12 {
        m.players[0].board.set(1, r, C::Red);
    }
    assert!(m.players[0].board.is_game_over());
    m.players[0].phase = PlayerPhase::Chaining;
    advance(&mut m);
    assert_eq!(m.players[0].phase, PlayerPhase::Ready);
    assert_eq!(m.result(), MatchResult::Ongoing);
}

#[test]
fn identical_state_and_actions_replay_whole_matches() {
    for seed in 0..20 {
        let mut a = MatchState::new(seed);
        let mut b = a.clone();
        for _ in 0..200 {
            if a.result() != MatchResult::Ongoing {
                break;
            }
            let actions = [action(&a, P1), action(&a, P2)];
            assert_eq!(
                a.step(actions[0], actions[1]),
                b.step(actions[0], actions[1])
            );
            assert_eq!(a, b);
        }
        assert_ne!(a.result(), MatchResult::Ongoing);
    }
}

#[test]
fn redistribution_is_deterministic_balanced_and_capped() {
    for seed in 0..30 {
        for id in [P1, P2] {
            let mut a = MatchState::new(seed);
            a.tick = 19;
            let p = &mut a.players[id.index()];
            p.garbage_drop_count = 7;
            for col in [0, 5] {
                for row in 0..14 {
                    p.board.set(col, row, C::Garbage);
                }
            }
            p.confirmed_garbage = 60;
            let mut b = a.clone();
            assert_eq!(advance(&mut a), advance(&mut b));
            assert_eq!(a, b);
            assert_eq!(a.result(), MatchResult::Ongoing);
            let p = &a.players[id.index()];
            assert_eq!(p.confirmed_garbage, 30);
            let heights: Vec<_> = (1..5).map(|c| p.board.column_height(c)).collect();
            assert_eq!(heights.iter().sum::<usize>(), 30);
            assert_eq!(heights.iter().filter(|&&h| h == 7).count(), 2);
            assert_eq!(heights.iter().filter(|&&h| h == 8).count(), 2);
            assert!(a.requires_action(id));
        }
    }
}

#[test]
fn exhausted_capacity_keeps_undropped_garbage_and_uses_death_cell() {
    let mut m = MatchState::new(3);
    for col in 0..6 {
        for row in 0..if col == 2 { 11 } else { 14 } {
            m.players[0].board.set(col, row, C::Garbage);
        }
    }
    assert!(!m.players[0].board.is_game_over());
    m.players[0].confirmed_garbage = 30;
    let mut replay = m.clone();
    let e = advance(&mut m);
    assert_eq!(e, advance(&mut replay));
    assert_eq!(m, replay);
    assert_eq!(e.players[0].dropped, 3);
    assert_eq!(m.players[0].confirmed_garbage, 27);
    assert!(m.players[0].board.is_game_over());
    assert_eq!(m.result(), MatchResult::Player2Win);
}

fn trapped_board() -> Board {
    let mut b = board();
    for col in [1, 3] {
        for row in 0..13 {
            b.set(col, row, C::Garbage);
        }
    }
    b
}

#[test]
fn double_rotation_is_tsu_only_and_preserves_other_actions() {
    let b = trapped_board();
    let piece = Piece::new(C::Red, C::Blue);
    let south = Placement::new(2, Orientation::South);
    let solo = enumerate_placements(&b, &piece);
    let tsu = enumerate_tsu_placements(&b, &piece);
    assert!(!solo.contains(&south));
    assert!(tsu.contains(&south));
    assert_eq!(
        tsu.into_iter().filter(|p| *p != south).collect::<Vec<_>>(),
        solo
    );
    let same_color = Piece::new(C::Red, C::Red);
    assert_eq!(
        enumerate_tsu_placements(&b, &same_color),
        enumerate_placements(&b, &same_color)
    );
}

#[test]
fn pvp_accepts_double_rotation_and_places_satellite_below_axis() {
    for id in [P1, P2] {
        let mut m = MatchState::new(0);
        // Find an existing deterministic mixed-color piece, without overriding RNG.
        let index = (0..100)
            .find(|&i| {
                let piece = m.piece_at(i);
                piece.axis_color != piece.satellite_color
            })
            .unwrap();
        m.players[id.index()].piece_index = index;
        m.players[id.index()].board = trapped_board();
        let piece = m.current_piece(id);
        let south = Placement::new(2, Orientation::South);
        assert!(m.legal_actions(id).contains(&south));
        let mut actions = [action(&m, P1), action(&m, P2)];
        actions[id.index()] = Some(south);
        let e = m.step(actions[0], actions[1]).unwrap();
        assert!(e.players[id.index()].placed);
        assert_eq!(m.players[id.index()].piece_index, index + 1);
        assert_eq!(m.players[id.index()].board.get(2, 0), piece.satellite_color);
        assert_eq!(m.players[id.index()].board.get(2, 1), piece.axis_color);
        assert_eq!(m.result(), MatchResult::Ongoing);
    }
}

#[test]
fn double_rotation_still_requires_reachability_and_landing_space() {
    let mut b = trapped_board();
    let piece = Piece::new(C::Red, C::Blue);
    let south = Placement::new(2, Orientation::South);
    // col0 is empty but unreachable across the 13-high wall at col1.
    assert!(!enumerate_tsu_placements(&b, &piece).contains(&Placement::new(0, Orientation::South)));
    for row in 0..11 {
        b.set(2, row, C::Garbage);
    }
    assert!(enumerate_tsu_placements(&b, &piece).contains(&south));
    // Preserve the existing restriction against landing the axis at row13.
    b.set(2, 11, C::Garbage);
    assert!(!enumerate_tsu_placements(&b, &piece).contains(&south));
    b.set(2, 12, C::Garbage);
    assert!(!enumerate_tsu_placements(&b, &piece).contains(&south));
}
