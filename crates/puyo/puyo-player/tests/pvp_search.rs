#![cfg(feature = "nn")]
use az_framework::mcts::InferenceProvider;
use puyo_core::{
    board::PuyoColor as C,
    piece::{Orientation, Placement},
    pvp::*,
    pvp_encoding::*,
};
use puyo_player::pvp_search::*;
use PlayerId::{Player1 as P1, Player2 as P2};
struct Mock {
    value: f32,
    action: Option<usize>,
}
impl InferenceProvider for Mock {
    fn infer(&self, _: &[f32], _: &[f32]) -> (Vec<f32>, f32) {
        let mut logits = vec![0.; 24];
        if let Some(a) = self.action {
            logits[a] = 100.;
        }
        (logits, self.value)
    }
}
fn config() -> PvpSearchConfig {
    PvpSearchConfig {
        simulations: 8,
        root_noise: false,
        ..Default::default()
    }
}
fn chain(s: &mut MatchState, id: PlayerId) {
    let b = &mut s.players[id.index()].board;
    for _ in 0..3 {
        b.drop_puyo(0, C::Blue);
    }
    for _ in 0..4 {
        b.drop_puyo(1, C::Red);
    }
    b.drop_puyo(1, C::Blue);
    s.players[id.index()].phase = PlayerPhase::Chaining;
}
#[test]
fn legal_policy_normalized_and_reproducible() {
    let s = MatchState::new(1);
    let p = Mock {
        value: 0.2,
        action: None,
    };
    let c = config();
    let a = PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap();
    let b = PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap();
    assert_eq!(a, b);
    assert!((a.policy.iter().sum::<f32>() - 1.).abs() < 1e-6);
    assert_eq!(a.visits.iter().sum::<u32>(), 8);
    for (i, legal) in pvp_valid_action_mask(&s, P1).iter().enumerate() {
        if !legal {
            assert_eq!(a.policy[i], 0.);
            assert_eq!(a.root_priors[i], 0.);
        }
    }
    assert!((a.value - 0.2).abs() < 1e-6);
    assert_eq!(a.select_action(1., 33), b.select_action(1., 33));
    assert!(s
        .legal_actions(P1)
        .contains(&a.select_action(0., 0).unwrap()));
}
#[test]
fn terminal_and_backup_use_fixed_perspective_without_score() {
    let p = Mock {
        value: 0.25,
        action: Some(8),
    };
    for (result, z) in [
        (MatchResult::Player1Win, 1.),
        (MatchResult::Player2Win, -1.),
        (MatchResult::Draw, 0.),
    ] {
        let mut s = MatchState::new(1);
        s.result = result;
        s.players[0].score = 999999;
        assert_eq!(
            PvpSearchV1::search(&s, P1, &p, &config(), 0).unwrap().value,
            z
        );
        assert_eq!(
            PvpSearchV1::search(&s, P2, &p, &config(), 0).unwrap().value,
            -z
        );
    }
    let mut s = MatchState::new(1);
    for r in 0..11 {
        s.players[0].board.set(2, r, C::Garbage);
    }
    let c = PvpSearchConfig {
        simulations: 1,
        ..config()
    };
    assert_eq!(PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap().value, -1.);
    assert_eq!(PvpSearchV1::search(&s, P2, &p, &c, 0).unwrap().value, 1.);
    for r in 0..11 {
        s.players[1].board.set(2, r, C::Garbage);
    }
    assert_eq!(PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap().value, 0.);
    let mut s = MatchState::new(3);
    let color = s.current_piece(P1).axis_color;
    for r in 0..3 {
        s.players[0].board.set(0, r, color);
    }
    let p = Mock {
        value: 0.25,
        action: Some(0),
    };
    let a = PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap();
    assert_eq!(a.value, 0.25);
    s.players[0].score = 1000000;
    assert_eq!(a, PvpSearchV1::search(&s, P1, &p, &c, 0).unwrap());
}
#[test]
fn opponent_baseline_is_pre_tick_and_shared_across_candidates() {
    use std::cell::RefCell;
    struct Spy(RefCell<Vec<(Vec<f32>, Vec<f32>)>>);
    impl InferenceProvider for Spy {
        fn infer(&self, b: &[f32], c: &[f32]) -> (Vec<f32>, f32) {
            self.0.borrow_mut().push((b.to_vec(), c.to_vec()));
            (vec![0.; 24], 0.)
        }
    }
    let s = MatchState::new(33);
    let spy = Spy(RefCell::new(Vec::new()));
    let decision = PreparedDecision::new(&s, P1, &spy, &MaskedArgmax).unwrap();
    assert_eq!(
        spy.0.borrow().as_slice(),
        &[(
            pvp_board_to_tensor_data(&s, P2),
            pvp_context_to_tensor_data(&s, P2)
        )]
    );
    let legal = s.legal_actions(P1);
    let a = decision.step_candidate(legal[0]).unwrap();
    let b = decision.step_candidate(legal[1]).unwrap();
    assert_eq!(a.players[1], b.players[1]);
    assert_ne!(a.players[0].board, b.players[0].board);
    assert_eq!(spy.0.borrow().len(), 1);
    assert_eq!(s.tick, 0);
}
#[test]
fn forced_ticks_handle_chains_drops_terminal_and_limits() {
    let p = Mock {
        value: 0.,
        action: None,
    };
    let mut s = MatchState::new(1);
    chain(&mut s, P1);
    assert!(!s.requires_action(P1));
    assert_eq!(
        advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 10).unwrap(),
        2
    );
    assert_eq!(s.players[0].piece_index, 0);
    assert_eq!(s.players[1].piece_index, 2);
    let mut s = MatchState::new(1);
    chain(&mut s, P2);
    assert_eq!(
        advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 10).unwrap(),
        0
    );
    let mut s = MatchState::new(1);
    chain(&mut s, P1);
    chain(&mut s, P2);
    assert_eq!(
        advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 10).unwrap(),
        2
    );
    assert_eq!(s.players[0].piece_index, 0);
    assert_eq!(s.players[1].piece_index, 0);
    let mut s = MatchState::new(1);
    s.players[0].confirmed_garbage = 6;
    assert_eq!(
        advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 10).unwrap(),
        1
    );
    assert_eq!(s.players[0].piece_index, 0);
    let mut s = MatchState::new(1);
    chain(&mut s, P1);
    assert!(advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 0).is_err());
    s.result = MatchResult::Draw;
    assert_eq!(
        advance_until_decision(&mut s, P1, &p, &MaskedArgmax, 0).unwrap(),
        0
    );
}
#[test]
fn independent_reproducible_noise_breaks_symmetry_and_streams_can_swap() {
    let p = Mock {
        value: 0.,
        action: None,
    };
    let mut c = config();
    c.root_noise = true;
    let mut different = 0;
    for seed in 0..24 {
        let s = MatchState::new(seed);
        assert_eq!(s.current_piece(P1), s.current_piece(P2));
        let seeds = [0, 1].map(|stream| derive_seed(seed, 0, stream, SeedPurpose::RootNoise));
        assert_ne!(seeds[0], seeds[1]);
        assert_eq!(seeds[0], derive_seed(seed, 0, 0, SeedPurpose::RootNoise));
        assert_ne!(seeds[0], derive_seed(seed, 0, 0, SeedPurpose::ActionSample));
        let a = PvpSearchV1::search(&s, P1, &p, &c, seeds[0]).unwrap();
        let b = PvpSearchV1::search(&s, P2, &p, &c, seeds[1]).unwrap();
        assert_ne!(a.root_priors, b.root_priors);
        assert_eq!(a, PvpSearchV1::search(&s, P2, &p, &c, seeds[0]).unwrap());
        assert_eq!(b, PvpSearchV1::search(&s, P1, &p, &c, seeds[1]).unwrap());
        let actions = [(&a, 0), (&b, 1)].map(|(r, stream)| {
            r.select_action(1., derive_seed(seed, 0, stream, SeedPurpose::ActionSample))
                .unwrap()
        });
        different += usize::from(actions[0] != actions[1]);
    }
    assert!(different > 0);
}
#[test]
fn forced_opponent_queries_use_each_pre_tick_snapshot() {
    use std::cell::RefCell;
    struct Spy(RefCell<Vec<Vec<f32>>>);
    impl InferenceProvider for Spy {
        fn infer(&self, _: &[f32], c: &[f32]) -> (Vec<f32>, f32) {
            self.0.borrow_mut().push(c.to_vec());
            (vec![0.; 24], 0.)
        }
    }
    let mut s = MatchState::new(19);
    chain(&mut s, P1);
    let mut expected = s.clone();
    let spy = Spy(RefCell::new(Vec::new()));
    let before = pvp_context_to_tensor_data(&s, P2);
    let opponent_action = MaskedArgmax
        .action(
            &Mock {
                value: 0.,
                action: None,
            },
            &expected,
            P2,
        )
        .unwrap();
    expected.step(None, opponent_action).unwrap();
    let second = pvp_context_to_tensor_data(&expected, P2);
    advance_until_decision(&mut s, P1, &spy, &MaskedArgmax, 10).unwrap();
    assert_eq!(*spy.0.borrow(), vec![before, second]);
}
#[test]
fn tsu_legal_mask_used() {
    let mut s = MatchState::new(0);
    s.players[0].piece_index = (0..100)
        .find(|&i| {
            let p = s.piece_at(i);
            p.axis_color != p.satellite_color
        })
        .unwrap();
    for col in [1, 3] {
        for r in 0..13 {
            s.players[0].board.set(col, r, C::Garbage);
        }
    }
    assert!(s
        .legal_actions(P1)
        .contains(&Placement::new(2, Orientation::South)));
    assert!(pvp_valid_action_mask(&s, P1)[10]);
}

#[test]
fn default_sixty_four_simulations_complete() {
    let s = MatchState::new(4);
    let result = PvpSearchV1::search(
        &s,
        P1,
        &Mock {
            value: 0.2,
            action: None,
        },
        &PvpSearchConfig::default(),
        44,
    )
    .unwrap();
    assert_eq!(result.visits.iter().sum::<u32>(), 64);
    assert!((result.policy.iter().sum::<f32>() - 1.).abs() < 1e-6);
}
