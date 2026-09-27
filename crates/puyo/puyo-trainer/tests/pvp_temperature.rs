use az_framework::mcts::InferenceProvider;
use puyo_core::{board::PuyoColor, placement::placement_to_index, pvp::*};
use puyo_player::pvp_search::*;
use puyo_trainer::{pvp_cli::Args, pvp_self_play::*};
use PlayerId::{Player1 as P1, Player2 as P2};
struct Mock;
impl InferenceProvider for Mock {
    fn infer(&self, _: &[f32], _: &[f32]) -> (Vec<f32>, f32) {
        (vec![0.; 24], 0.)
    }
}
#[test]
fn per_player_boundaries_and_sampling_vs_argmax() {
    let schedule = TemperatureSchedule::PieceIndex(20);
    let mut state = MatchState::new(1);
    state.players[0].piece_index = 19;
    assert_eq!(schedule.temperature(&state, P1, 1.), 1.);
    state.players[0].piece_index = 20;
    assert_eq!(schedule.temperature(&state, P1, 1.), 0.);
    state.tick = 100;
    state.players[0].piece_index = 25;
    state.players[1].piece_index = 10;
    assert_eq!(schedule.temperature(&state, P1, 1.), 0.);
    assert_eq!(schedule.temperature(&state, P2, 1.), 1.);
    let mut visits = [0; 24];
    visits[0] = 3;
    visits[4] = 2;
    let r = PvpSearchResult {
        visits,
        policy: [0.; 24],
        root_priors: [0.; 24],
        value: 0.,
    };
    let mut sampled_other = false;
    for seed in 0..100 {
        assert_eq!(
            placement_to_index(
                &r.select_action(schedule.temperature(&state, P1, 1.), seed)
                    .unwrap()
            ),
            0
        );
        let sampled = r
            .select_action(schedule.temperature(&state, P2, 1.), seed)
            .unwrap();
        assert_eq!(sampled, r.select_action(1., seed).unwrap());
        sampled_other |= placement_to_index(&sampled) == 4;
        assert_eq!(
            placement_to_index(
                &r.select_action(schedule.temperature(&state, P2, 0.), seed)
                    .unwrap()
            ),
            0
        );
    }
    assert!(sampled_other);
}
#[test]
fn chain_ticks_do_not_advance_piece_schedule() {
    let mut state = MatchState::new(4);
    for player in &mut state.players {
        player.piece_index = 19;
        player.phase = PlayerPhase::Chaining;
        for row in 0..4 {
            player.board.set(0, row, PuyoColor::Red);
        }
    }
    state.tick = 99;
    state.step(None, None).unwrap();
    assert_eq!(state.tick, 100);
    for id in [P1, P2] {
        assert_eq!(state.players[id.index()].piece_index, 19);
        assert_eq!(
            TemperatureSchedule::PieceIndex(20).temperature(&state, id, 1.),
            1.
        );
        assert_eq!(
            TemperatureSchedule::LegacyTick(100).temperature(&state, id, 1.),
            0.
        );
    }
}
#[test]
fn cli_default_legacy_and_conflict() {
    let allowed = ["--temperature-drop-tick", "--temperature-drop-piece"];
    let parse = |s: &[&str]| Args::parse_from(s.iter().map(|x| x.to_string()), &allowed).unwrap();
    assert_eq!(
        parse(&[]).temperature_schedule().unwrap(),
        TemperatureSchedule::PieceIndex(20)
    );
    assert_eq!(
        parse(&["--temperature-drop-tick", "100"])
            .temperature_schedule()
            .unwrap(),
        TemperatureSchedule::LegacyTick(100)
    );
    assert!(parse(&[
        "--temperature-drop-tick",
        "100",
        "--temperature-drop-piece",
        "20"
    ])
    .temperature_schedule()
    .is_err());
}
#[test]
fn diagnostics_count_schedule_and_replay() {
    let c = SelfPlayConfig {
        temperature_schedule: TemperatureSchedule::PieceIndex(2),
        max_ticks: 5,
        search: PvpSearchConfig {
            simulations: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let a = play_match(&Mock, 42, &c).unwrap();
    assert_eq!(a, play_match(&Mock, 42, &c).unwrap());
    assert_eq!(a.metrics.sampled_decisions, [2, 2]);
    assert!(a.metrics.argmax_decisions.iter().all(|&n| n > 0));
    let a = play_match(
        &Mock,
        42,
        &SelfPlayConfig {
            temperature: 0.,
            ..c
        },
    )
    .unwrap();
    assert_eq!(a.metrics.sampled_decisions, [0, 0]);
}
