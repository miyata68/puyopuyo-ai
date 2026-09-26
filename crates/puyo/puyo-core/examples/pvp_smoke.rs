//! Deterministic random-vs-random diagnostic; tick-limit draws are harness-only.
use puyo_core::board::PuyoColor;
use puyo_core::pvp::{MatchResult, MatchState, PlayerId, PlayerPhase};
fn random(x: &mut u64) -> u64 {
    *x = x
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *x ^ (*x >> 29)
}
fn check_non_death_overflow() {
    for seed in 0..100 {
        let mut m = MatchState::new(seed);
        for row in 0..14 {
            m.players[0].board.set(0, row, PuyoColor::Garbage);
        }
        m.players[0].confirmed_garbage = 30;
        let mut replay = m.clone();
        let action = m.legal_actions(PlayerId::Player2)[0];
        let event = m.step(None, Some(action)).expect("overflow redistribution");
        assert_eq!(event, replay.step(None, Some(action)).unwrap());
        assert_eq!(m, replay);
        assert_eq!(event.players[0].dropped, 30);
        assert_eq!(m.players[0].confirmed_garbage, 0);
        assert_eq!(m.result(), MatchResult::Ongoing);
        for col in 1..6 {
            assert_eq!(m.players[0].board.column_height(col), 6);
        }
    }
    println!("OVERFLOW checks=100 ongoing=100 redistributed_without_loss=true deterministic=true");
}
fn main() {
    check_non_death_overflow();
    let mut wins = [0u32; 3];
    let mut limited = 0;
    let mut attack = 0;
    let mut offset = 0;
    let mut dropped = 0;
    let mut during_chain = 0;
    let mut multi_chain = 0;
    let mut max_tick = 0;
    for seed in 0..100 {
        let mut m = MatchState::new(seed);
        let mut rng = seed + 12345;
        let mut streak = [0; 2];
        while m.result() == MatchResult::Ongoing && m.tick < 2000 {
            let mut actions = [None; 2];
            for id in [PlayerId::Player1, PlayerId::Player2] {
                let legal = m.legal_actions(id);
                if m.requires_action(id) {
                    assert!(!legal.is_empty());
                    actions[id.index()] = Some(legal[random(&mut rng) as usize % legal.len()]);
                }
            }
            let chaining = m
                .players
                .each_ref()
                .map(|p| p.phase == PlayerPhase::Chaining);
            let e = m
                .step(actions[0], actions[1])
                .expect("legal random actions");
            for i in 0..2 {
                attack += e.players[i].generated_attack;
                offset += e.players[i].offset;
                dropped += u64::from(e.players[i].dropped);
                if chaining[1 - i] && e.players[i].placed {
                    during_chain += 1;
                    streak[i] += 1;
                    if streak[i] == 2 {
                        multi_chain += 1;
                    }
                } else {
                    streak[i] = 0;
                }
            }
        }
        max_tick = max_tick.max(m.tick);
        let result = if m.result() == MatchResult::Ongoing {
            limited += 1;
            MatchResult::Draw
        } else {
            m.result()
        };
        wins[match result {
            MatchResult::Player1Win => 0,
            MatchResult::Player2Win => 1,
            _ => 2,
        }] += 1;
        println!(
            "seed={seed} ticks={} result={result:?} pieces={:?} max_chain={:?}",
            m.tick,
            m.players.each_ref().map(|p| p.piece_index),
            m.players.each_ref().map(|p| p.max_chain)
        );
    }
    println!("SUMMARY games=100 p1={} p2={} draw={} tick_limit_draws={limited} max_tick={max_tick} generated={attack} offset={offset} dropped={dropped} placements_during_chain={during_chain} multiple_placements_during_one_chain={multi_chain}",wins[0],wins[1],wins[2]);
    assert!(attack > 0 && offset > 0 && dropped > 0 && multi_chain > 0);
}
