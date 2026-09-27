use burn::{
    module::Module,
    prelude::*,
    record::{FullPrecisionSettings, Record},
};
use puyo_core::pvp_encoding::*;
use puyo_nn::pvp_model::PvpPuyoNetConfig;
use puyo_trainer::{pvp_data::*, pvp_metrics::*, pvp_training::losses};
fn sample() -> PvpAlphaZeroSample {
    let mut s = PvpAlphaZeroSample {
        board_data: vec![0.; BOARD_SIZE],
        context_data: vec![0.; CONTEXT_SIZE],
        improved_policy: vec![0.; NUM_ACTIONS],
        value_target: 1.,
        perspective: 0,
        legal_mask: vec![false; NUM_ACTIONS],
    };
    s.improved_policy[0] = 0.5;
    s.improved_policy[1] = 0.5;
    s.legal_mask[0] = true;
    s.legal_mask[1] = true;
    s
}
#[test]
fn policy_metrics_known_distributions_legal_zero_and_illegal_logits() {
    type B = burn::backend::NdArray;
    let device = Default::default();
    let mut s = sample();
    let mut logits = vec![1000.; NUM_ACTIONS];
    logits[0] = 0.;
    logits[1] = 0.;
    let m = metrics_from_predictions(&logits, &[0.], &[s.clone()]).unwrap();
    assert!((m.policy_ce - 2f64.ln()).abs() < 1e-10);
    assert!((m.target_entropy - 2f64.ln()).abs() < 1e-10);
    assert_eq!(m.policy_kl, 0.);
    assert_eq!(m.value_mse, 1.);
    s.legal_mask[2] = true;
    logits[2] = 0.;
    let m = metrics_from_predictions(&logits, &[0.5], &[s.clone()]).unwrap();
    assert!((m.policy_ce - 3f64.ln()).abs() < 1e-10);
    assert!((m.policy_kl - (1.5f64).ln()).abs() < 1e-10);
    assert_eq!(m.value_mse, 0.25);
    logits[0] = 1.3;
    logits[1] = -0.5;
    let m = metrics_from_predictions(&logits, &[0.5], &[s.clone()]).unwrap();
    assert!((m.policy_ce - m.target_entropy - m.policy_kl).abs() < 1e-10);
    let (ce, _) = losses::<B>(
        Tensor::<B, 1>::from_floats(logits.as_slice(), &device).reshape([1, NUM_ACTIONS]),
        Tensor::zeros([1, 1], &device),
        &[s.clone()],
        &device,
    );
    assert!((ce.into_scalar() as f64 - m.policy_ce).abs() < 1e-5);
    s.improved_policy.fill(0.);
    s.improved_policy[0] = 1.;
    let m = metrics_from_predictions(&logits, &[1.], &[s.clone()]).unwrap();
    assert_eq!(m.target_entropy, 0.);
    assert_eq!(m.policy_ce, m.policy_kl);
    logits[0] = f32::NAN;
    assert!(metrics_from_predictions(&logits, &[1.], &[s]).is_err());
}
#[test]
fn evaluation_snapshot_preserves_all_parameters_and_bn_state_and_batches_correctly() {
    type B = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = Default::default();
    let net = PvpPuyoNetConfig::new()
        .with_residual_channels(8)
        .with_num_residual_blocks(1)
        .init::<B>(&device);
    let mut samples = vec![sample(); 3];
    samples[1].board_data[0] = 1.;
    samples[2].board_data[85] = 1.;
    samples[2].value_target = -1.;
    // Populate running BN state via a training forward before evaluating.
    let (p, v) = forward_samples(&net, &samples, &device);
    let _ = (p.into_data(), v.into_data());
    let before = serde_json::to_value(
        net.clone()
            .into_record()
            .into_item::<FullPrecisionSettings>(),
    )
    .unwrap();
    let raw_samples = samples.clone();
    let a = evaluate_snapshot(&net, &samples, &[0, 1, 2], 2, &device).unwrap();
    let b = evaluate_snapshot(&net, &samples, &[0, 1, 2], 1, &device).unwrap();
    let after = serde_json::to_value(
        net.clone()
            .into_record()
            .into_item::<FullPrecisionSettings>(),
    )
    .unwrap();
    assert_eq!(before, after);
    assert_eq!(samples, raw_samples);
    assert_eq!(a.samples, 3);
    assert!((a.policy_ce - b.policy_ce).abs() < 1e-5);
    assert!((a.value_mse - b.value_mse).abs() < 1e-5);
    assert!(a.policy_ce.is_finite() && a.policy_kl.is_finite() && a.value_mse.is_finite());
    assert!(evaluate_snapshot(&net, &samples, &[], 2, &device).is_err());
}
