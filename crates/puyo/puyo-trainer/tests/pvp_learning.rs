use az_framework::mcts::InferenceProvider;
use burn::prelude::*;
use puyo_core::{pvp::*, pvp_encoding::*};
use puyo_nn::pvp_model::PvpPuyoNetConfig;
use puyo_player::pvp_search::PvpSearchConfig;
use puyo_trainer::{pvp_data::*, pvp_self_play::*, pvp_training::*};
use PlayerId::{Player1 as P1, Player2 as P2};
fn record(id: PlayerId) -> MoveRecord {
    let s = MatchState::new(1);
    let mask = pvp_valid_action_mask(&s, id);
    let n = mask.iter().filter(|&&v| v).count() as f32;
    MoveRecord {
        board_data: pvp_board_to_tensor_data(&s, id),
        context_data: pvp_context_to_tensor_data(&s, id),
        improved_policy: mask.map(|v| if v { 1. / n } else { 0. }).to_vec(),
        perspective: id,
        legal_mask: mask.to_vec(),
    }
}
#[test]
fn targets_are_only_outcomes_and_truncation_is_discarded() {
    for (r, z) in [
        (MatchResult::Player1Win, [1., -1.]),
        (MatchResult::Player2Win, [-1., 1.]),
        (MatchResult::Draw, [0., 0.]),
    ] {
        let samples = finish_records(vec![record(P1), record(P2)], r);
        assert_eq!(
            samples.iter().map(|s| s.value_target).collect::<Vec<_>>(),
            z
        );
        for s in samples {
            s.validate().unwrap();
        }
    }
    assert!(finish_records(vec![record(P1)], MatchResult::Ongoing).is_empty());
}
#[test]
fn four_color_permutations_preserve_garbage_scalars_and_targets() {
    let sample = finish_records(vec![record(P1)], MatchResult::Player1Win).remove(0);
    let mut sample = sample;
    sample.board_data[0] = 1.;
    sample.board_data[5 * 84] = 1.;
    sample.board_data[4 * 84] = 1.;
    sample.board_data[9 * 84] = 1.;
    for (i, x) in sample.context_data[48..].iter_mut().enumerate() {
        *x = i as f32;
    }
    for perm in puyo_trainer::data::all_color_permutations(4) {
        let mut a = sample.clone();
        augment_colors(&mut a, &perm).unwrap();
        assert_eq!(a.board_data[perm[0] * 84], 1.);
        assert_eq!(a.board_data[(5 + perm[0]) * 84], 1.);
        assert_eq!(
            &a.board_data[4 * 84..5 * 84],
            &sample.board_data[4 * 84..5 * 84]
        );
        assert_eq!(
            &a.board_data[9 * 84..10 * 84],
            &sample.board_data[9 * 84..10 * 84]
        );
        assert_eq!(&a.context_data[48..], &sample.context_data[48..]);
        for block in 0..12 {
            for (old, &new) in perm.iter().enumerate() {
                assert_eq!(
                    a.context_data[block * 4 + new],
                    sample.context_data[block * 4 + old]
                );
            }
        }
        assert_eq!(a.value_target, sample.value_target);
        assert_eq!(a.improved_policy, sample.improved_policy);
    }
}
#[test]
fn metadata_dataset_roundtrip_and_reject_solo() {
    let dir = std::env::temp_dir().join(format!("pvp-data-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dataset.bin");
    let data = PvpAlphaZeroDataset {
        metadata: Default::default(),
        samples: finish_records(vec![record(P1)], MatchResult::Draw),
    };
    data.save(&path).unwrap();
    assert_eq!(data, PvpAlphaZeroDataset::load(&path).unwrap());
    std::fs::write(&path, b"solo dataset").unwrap();
    assert!(PvpAlphaZeroDataset::load(&path).is_err());
    let mut bad = data.clone();
    bad.metadata.context_size = 99;
    assert!(bad.save(&path).is_err());
    let path = dir.join("model").to_string_lossy().into_owned();
    std::fs::write(format!("{path}.config.json"), "{\"game_config\":{}}").unwrap();
    assert!(PvpModelMetadata::read(&path)
        .unwrap_err()
        .contains("incompatible model kind"));
    let metadata = PvpModelMetadata::new(PvpPuyoNetConfig::new());
    metadata.write(&path).unwrap();
    assert_eq!(
        PvpModelMetadata::read(&path).unwrap().dimensions,
        PvpDataMetadata::default()
    );
}
struct Mock;
impl InferenceProvider for Mock {
    fn infer(&self, _: &[f32], _: &[f32]) -> (Vec<f32>, f32) {
        (vec![0.; 24], 0.)
    }
}
#[test]
fn self_play_replay_symmetry_and_truncation() {
    let config = SelfPlayConfig {
        search: PvpSearchConfig {
            simulations: 8,
            ..Default::default()
        },
        max_ticks: 200,
        ..Default::default()
    };
    let mut broken = 0;
    let mut samples = [0; 2];
    for seed in 0..8 {
        let a = play_match(&Mock, seed, &config).unwrap();
        let b = play_match(&Mock, seed, &config).unwrap();
        assert_eq!(a, b);
        broken += usize::from(a.metrics.symmetry_break_tick.is_some());
        for s in &a.samples {
            s.validate().unwrap();
            samples[s.perspective as usize] += 1;
        }
    }
    assert!(broken > 0 && samples[0] > 0 && samples[1] > 0);
    let config = SelfPlayConfig {
        max_ticks: 1,
        ..config
    };
    let a = play_match(&Mock, 42, &config).unwrap();
    assert!(a.metrics.truncated);
    assert!(a.samples.is_empty());
    assert_eq!(a.final_state.result(), MatchResult::Ongoing);
}
#[test]
fn mirrored_assignment_swaps_first_tick_actions() {
    let c = SelfPlayConfig {
        max_ticks: 1,
        search: PvpSearchConfig {
            simulations: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let a = play_match(&Mock, 3, &c).unwrap();
    let b = play_match(
        &Mock,
        3,
        &SelfPlayConfig {
            streams: [1, 0],
            ..c
        },
    )
    .unwrap();
    assert_eq!(
        a.metrics.actions[0],
        [b.metrics.actions[0][1], b.metrics.actions[0][0]]
    );
}
#[test]
fn nn_shapes_bounded_values_and_training_losses() {
    type B = burn::backend::NdArray;
    let d = Default::default();
    let config = PvpPuyoNetConfig::new()
        .with_residual_channels(8)
        .with_num_residual_blocks(1);
    let net = config.init::<B>(&d);
    let (policy, value) = net.forward(
        Tensor::zeros([2, 10, 14, 6], &d),
        Tensor::zeros([2, 68], &d),
    );
    assert_eq!(policy.dims(), [2, 24]);
    assert_eq!(value.dims(), [2, 1]);
    assert!(value
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));
    let mut samples = finish_records(vec![record(P1), record(P2)], MatchResult::Player1Win);
    for s in &mut samples {
        s.improved_policy.fill(0.);
        s.improved_policy[0] = 1.;
        s.legal_mask.fill(true);
    }
    let (p, v) = losses::<B>(
        Tensor::zeros([2, 24], &d),
        Tensor::zeros([2, 1], &d),
        &samples,
        &d,
    );
    assert!((p.into_scalar() - (24f32).ln()).abs() < 1e-5);
    assert_eq!(v.into_scalar(), 1.);
}
