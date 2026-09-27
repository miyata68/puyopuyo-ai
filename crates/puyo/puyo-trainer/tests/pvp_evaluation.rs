use az_framework::mcts::InferenceProvider;
use puyo_core::{board::PuyoColor as C, placement::placement_to_index, pvp::*, pvp_encoding::*};
use puyo_player::pvp_search::*;
use puyo_trainer::pvp_evaluation::*;
use std::cell::RefCell;
use PlayerId::{Player1 as P1, Player2 as P2};
type Observation = (Vec<f32>, Vec<f32>);
struct Spy {
    action: usize,
    value: f32,
    inputs: RefCell<Vec<Observation>>,
}
impl Spy {
    fn new(action: usize, value: f32) -> Self {
        Self {
            action,
            value,
            inputs: RefCell::new(vec![]),
        }
    }
    fn take(&self) -> Vec<Observation> {
        self.inputs.take()
    }
}
impl InferenceProvider for Spy {
    fn infer(&self, board: &[f32], context: &[f32]) -> (Vec<f32>, f32) {
        self.inputs
            .borrow_mut()
            .push((board.to_vec(), context.to_vec()));
        let mut logits = vec![0.; NUM_ACTIONS];
        logits[self.action] = 100.;
        (logits, self.value)
    }
}
fn observation(s: &MatchState, id: PlayerId) -> Observation {
    (
        pvp_board_to_tensor_data(s, id),
        pvp_context_to_tensor_data(s, id),
    )
}
#[test]
fn separate_providers_at_root_leaf_and_forced_ticks_in_both_directions() {
    let a = Spy::new(0, 0.4);
    let b = Spy::new(4, -0.3);
    for id in [P1, P2] {
        let other = opponent(id);
        let mut s = MatchState::new(4);
        let color = s.current_piece(id).axis_color;
        // Choosing action 0 clears this column, forcing opponent-only decisions.
        for row in 0..3 {
            s.players[id.index()].board.set(0, row, color);
        }
        let config = PvpSearchConfig {
            simulations: 1,
            root_noise: false,
            ..Default::default()
        };
        let r = PvpSearchV1::search_with_providers(&s, id, &a, &b, &config, 0).unwrap();
        let mut expected = s.clone();
        let mut actions = [None; 2];
        actions[id.index()] = Some(r.select_action(0., 0).unwrap());
        actions[other.index()] = Some(puyo_core::placement::index_to_placement(4, COLS));
        let mut expected_b = vec![observation(&expected, other)];
        expected.step(actions[0], actions[1]).unwrap();
        assert!(!expected.requires_action(id));
        while !expected.requires_action(id) && expected.result() == MatchResult::Ongoing {
            let mut actions = [None; 2];
            if expected.requires_action(other) {
                expected_b.push(observation(&expected, other));
                actions[other.index()] = Some(puyo_core::placement::index_to_placement(4, COLS));
            }
            expected.step(actions[0], actions[1]).unwrap();
        }
        if expected.requires_action(other) {
            expected_b.push(observation(&expected, other));
        }
        assert_eq!(
            a.take(),
            vec![observation(&s, id), observation(&expected, id)]
        );
        assert_eq!(b.take(), expected_b);
        assert!((r.value - 0.4).abs() < 1e-6);
    }
    let s = MatchState::new(4);
    let c = PvpSearchConfig {
        simulations: 1,
        root_noise: false,
        ..Default::default()
    };
    let r = PvpSearchV1::search_with_providers(&s, P2, &b, &a, &c, 99).unwrap();
    assert!((r.value + 0.3).abs() < 1e-6);
    assert_eq!(b.take()[0], observation(&s, P2));
    assert_eq!(a.take()[0], observation(&s, P1));
}
#[test]
fn actual_searches_share_pre_tick_root_without_actual_action_leak_or_noise() {
    let a = Spy::new(0, 0.2);
    let b = Spy::new(4, -0.2);
    let mut s = MatchState::new(3);
    s.players[0].board.set(5, 0, C::Garbage);
    let before = s.clone();
    let c = EvalConfig {
        simulations: 4,
        max_ticks: 1,
    };
    assert!(!c.search_config().root_noise);
    let actual = evaluation_actions(&s, [&a, &b], &c).unwrap();
    let (actual_a_inputs, actual_b_inputs) = (a.take(), b.take());
    let r1 = PvpSearchV1::search_with_providers(&s, P1, &a, &b, &c.search_config(), 111).unwrap();
    let r2 = PvpSearchV1::search_with_providers(&s, P2, &b, &a, &c.search_config(), 999).unwrap();
    assert_eq!(
        actual,
        [
            Some(r1.select_action(0., 42).unwrap()),
            Some(r2.select_action(0., 43).unwrap())
        ]
    );
    assert_eq!(actual_a_inputs, a.take());
    assert_eq!(actual_b_inputs, b.take());
    assert_eq!(s, before);
    assert_ne!(actual[0], actual[1]);
    assert_eq!(
        r1,
        PvpSearchV1::search_with_providers(&s, P1, &a, &b, &c.search_config(), 333).unwrap()
    );
}
#[test]
fn paired_assignment_replay_and_truncation() {
    let a = Spy::new(0, 0.);
    let b = Spy::new(4, 0.);
    let c = EvalConfig {
        simulations: 2,
        max_ticks: 1,
    };
    let pair = evaluate_pair(&a, &b, 1, &c).unwrap();
    assert_eq!(pair, evaluate_pair(&a, &b, 1, &c).unwrap());
    assert_eq!(pair.games[0].model_a_seat, P1);
    assert_eq!(pair.games[1].model_a_seat, P2);
    assert_eq!(pair.games[0].seed, pair.games[1].seed);
    assert_eq!(pair.games[0].actions[0], [Some(0), Some(4)]);
    assert_eq!(pair.games[1].actions[0], [Some(4), Some(0)]);
    let mut summary = EvalSummary::default();
    summary.add_pair(&pair);
    assert_eq!(summary.a.truncated, 2);
    assert_eq!(summary.a.draws, 0);
    assert_eq!(summary.a.score_rate(), None);
    assert_eq!(summary.containing_truncation, 1);
    assert_eq!(summary.containing_draw, 0);
    assert_eq!(summary.a_as_seat[0].truncated, 1);
    assert_eq!(summary.a_as_seat[1].truncated, 1);
    // Independently verify max-visit selection for both seats.
    assert_eq!(
        placement_to_index(
            &evaluation_actions(&MatchState::new(1), [&a, &b], &c).unwrap()[0].unwrap()
        ),
        0
    );
}
#[test]
fn paired_summary_counts_model_outcomes_and_excludes_truncation_from_score() {
    let a = Spy::new(0, 0.);
    let b = Spy::new(4, 0.);
    let template = evaluate_pair(
        &a,
        &b,
        0,
        &EvalConfig {
            simulations: 1,
            max_ticks: 0,
        },
    )
    .unwrap();
    let mut summary = EvalSummary::default();
    for outcomes in [
        [Some(1), Some(1)],
        [Some(-1), Some(-1)],
        [Some(1), Some(-1)],
        [Some(0), Some(1)],
        [Some(0), None],
    ] {
        let mut pair = template.clone();
        for (game, outcome) in pair.games.iter_mut().zip(outcomes) {
            game.result = match outcome {
                None => MatchResult::Ongoing,
                Some(0) => MatchResult::Draw,
                Some(z) if (z == 1) == (game.model_a_seat == P1) => MatchResult::Player1Win,
                _ => MatchResult::Player2Win,
            };
        }
        summary.add_pair(&pair);
    }
    assert_eq!(
        summary.a,
        ScoreCounts {
            wins: 4,
            losses: 3,
            draws: 2,
            truncated: 1
        }
    );
    assert_eq!(summary.a.score(), 5.);
    assert_eq!(summary.a.score_rate(), Some(5. / 9.));
    assert_eq!(
        (summary.a_wins_both, summary.b_wins_both, summary.split),
        (1, 1, 1)
    );
    assert_eq!(
        (summary.containing_draw, summary.containing_truncation),
        (2, 1)
    );
}
