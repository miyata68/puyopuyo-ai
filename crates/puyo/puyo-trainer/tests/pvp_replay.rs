use puyo_core::pvp_encoding::*;
use puyo_player::pvp_search::SearchRng;
use puyo_trainer::{pvp_cli::Args, pvp_data::*, pvp_replay::*};
use std::collections::BTreeSet;
fn sample(index: usize) -> PvpAlphaZeroSample {
    let mut s = PvpAlphaZeroSample {
        board_data: vec![0.; BOARD_SIZE],
        context_data: vec![0.; CONTEXT_SIZE],
        improved_policy: vec![0.; NUM_ACTIONS],
        value_target: 1.,
        perspective: 0,
        legal_mask: vec![true; NUM_ACTIONS],
    };
    s.improved_policy[0] = 1.;
    s.board_data[0] = 1.;
    s.context_data[48] = index as f32;
    s
}
fn paths() -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("pvp-replay-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    (0..3)
        .map(|source| {
            let path = dir.join(format!("iter{source}.bin"));
            PvpAlphaZeroDataset {
                metadata: Default::default(),
                samples: (0..1000).map(|i| sample(source * 1000 + i)).collect(),
            }
            .save(&path)
            .unwrap();
            path.to_str().unwrap().to_string()
        })
        .collect()
}
fn ids(samples: &[PvpAlphaZeroSample]) -> BTreeSet<usize> {
    samples
        .iter()
        .map(|s| s.context_data[48] as usize)
        .collect()
}
#[test]
fn replay_union_stable_split_no_leakage_or_validation_augmentation() {
    let paths = paths();
    let pool = ReplayPool::load(&paths, 0.1, 42).unwrap();
    assert_eq!(pool.train().len() + pool.validation().len(), 3000);
    let fraction = pool.validation().len() as f64 / 3000.;
    assert!((fraction - 0.1).abs() < 0.03);
    let train = ids(pool.train());
    let val = ids(pool.validation());
    assert!(train.is_disjoint(&val));
    assert_eq!(
        train,
        ids(ReplayPool::load(&paths, 0.1, 42).unwrap().train())
    );
    let shifted = ReplayPool::load(&paths[1..], 0.1, 42).unwrap();
    assert_eq!(
        train
            .iter()
            .copied()
            .filter(|&i| i >= 1000)
            .collect::<BTreeSet<_>>(),
        ids(shifted.train())
    );
    assert_eq!(
        val.iter()
            .copied()
            .filter(|&i| i >= 1000)
            .collect::<BTreeSet<_>>(),
        ids(shifted.validation())
    );
    let before_val = pool.validation().to_vec();
    let before_train = pool.train().to_vec();
    let batch = pool
        .training_batch(5000, &mut SearchRng::new(9), &[vec![1, 2, 3, 0]])
        .unwrap();
    assert!(ids(&batch).is_subset(&train));
    assert!(ids(&batch).is_disjoint(&val));
    assert!(batch
        .iter()
        .all(|s| s.board_data[84] == 1. && s.board_data[0] == 0.));
    assert_eq!(before_val, pool.validation());
    assert_eq!(before_train, pool.train());
    let subset = fixed_subset(pool.train().len(), 20, 9);
    assert_eq!(subset, fixed_subset(pool.train().len(), 20, 9));
    assert_eq!(subset.len(), 20);
    assert_eq!(fixed_subset(3, 20, 9), vec![0, 1, 2]);
}
#[test]
fn replay_rejects_bad_metadata_duplicate_paths_fractions_and_empty_splits() {
    let dir = std::env::temp_dir().join(format!("pvp-replay-errors-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("good.bin").to_str().unwrap().to_string();
    let mut data = PvpAlphaZeroDataset {
        metadata: Default::default(),
        samples: (0..100).map(sample).collect(),
    };
    data.save(&path).unwrap();
    for fraction in [0., 0.5, 1., -0.1, f64::NAN] {
        assert!(ReplayPool::load(std::slice::from_ref(&path), fraction, 42).is_err());
    }
    let alias = dir.join(".").join("good.bin").to_str().unwrap().to_string();
    assert!(ReplayPool::load(&[path.clone(), alias], 0.1, 42)
        .err()
        .unwrap()
        .contains("duplicate"));
    data.metadata.context_size = 99;
    let bad = dir.join("bad.bin").to_str().unwrap().to_string();
    let mut bytes = b"PUYOPVP1".to_vec();
    bytes.extend(bincode::serialize(&data).unwrap());
    std::fs::write(&bad, bytes).unwrap();
    assert!(ReplayPool::load(&[path.clone(), bad], 0.1, 42)
        .err()
        .unwrap()
        .contains("metadata"));
    data.metadata = Default::default();
    data.samples.truncate(1);
    data.save(&path).unwrap();
    assert!(ReplayPool::load(&[path], 0.1, 42)
        .err()
        .unwrap()
        .contains("empty"));
}
#[test]
fn replay_cli_and_hash_are_fixed() {
    assert_eq!(split_hash("iter002", 1, 42), 262238462243500471);
    let parse = |s: &[&str]| {
        Args::parse_from(s.iter().map(|x| x.to_string()), &["--data", "--data-paths"]).unwrap()
    };
    assert_eq!(parse(&["--data", "x"]).data_paths().unwrap(), ["x"]);
    assert_eq!(
        parse(&["--data-paths", " a, b , c "]).data_paths().unwrap(),
        ["a", "b", "c"]
    );
    assert!(parse(&["--data", "x", "--data-paths", "y"])
        .data_paths()
        .is_err());
    assert!(parse(&["--data-paths", "a, a"]).data_paths().is_err());
    assert!(parse(&["--data-paths", "a,"]).data_paths().is_err());
    assert_ne!(split_hash("iter002", 1, 42), split_hash("iter002", 1, 43));
    assert_ne!(split_hash("iter002", 1, 42), split_hash("iter002", 2, 42));
}
