use burn::backend::Autodiff;
use burn::backend::Cuda;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

use az_framework::value_transform::value_transform;
use puyo_core::config::GameConfig;
use puyo_core::rand::time_seed;
use puyo_nn::model::PuyoNetConfig;
use puyo_trainer::data::{AlphaZeroDataset, Dataset};
use serde::{Deserialize, Serialize};

type TrainBackend = Autodiff<Cuda<f32>>;
type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

const MODEL_PATH: &str = "artifacts/puyo_model";
const BATCH_SIZE: usize = 512;
const NUM_EPOCHS: usize = 50;
const LR_MAX: f64 = 0.1;
const LR_MIN: f64 = 1e-3;
const EARLY_STOPPING_PATIENCE: usize = 5;

// AlphaZero-specific training parameters (step-based)
const AZ_NUM_STEPS: usize = 1000;
const ACCUM_STEPS: usize = 4; // gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS = 2048
                              // Global step LR schedule (AlphaZero-style 4-stage drop, default).
                              // Overridable via `--lr-stages "threshold:lr,threshold:lr,..."` CLI flag.
const AZ_LR_STAGES_DEFAULT: [(usize, f64); 4] = [
    (0, 0.2),        // step 0~10k:  LR = 0.2
    (10000, 0.02),   // step 10k~30k: LR = 0.02
    (30000, 0.002),  // step 30k~50k: LR = 0.002
    (50000, 0.0002), // step 50k~:    LR = 0.0002
];
const TRAIN_SPLIT_RATIO: f64 = 0.9;
const VALUE_LOSS_WEIGHT: f32 = 0.5;
const VALUE_SCALE: f32 = 15.0;

/// MSE loss between predicted value and transformed targets.
fn value_mse_loss<B: Backend>(
    value: Tensor<B, 2>,
    targets: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let batch_size = value.dims()[0];
    let transformed: Vec<f32> = targets
        .iter()
        .map(|&v| value_transform(v, VALUE_SCALE))
        .collect();
    let target_tensor =
        Tensor::<B, 1>::from_floats(transformed.as_slice(), device).reshape([batch_size, 1]);
    let diff = value - target_tensor;
    diff.clone().mul(diff).mean()
}

fn cosine_lr(epoch: usize, total_epochs: usize) -> f64 {
    LR_MIN
        + 0.5
            * (LR_MAX - LR_MIN)
            * (1.0 + (std::f64::consts::PI * epoch as f64 / total_epochs as f64).cos())
}

/// Cross-entropy loss for policy: -sum(target * log_softmax(logits)) / batch_size.
/// `targets` can be one-hot (u8 index) or soft (f32 distribution).
fn cross_entropy_loss_hard<B: Backend>(
    logits: Tensor<B, 2>,
    targets: &[u8],
    device: &B::Device,
    num_actions: usize,
) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];
    let max_logits = logits.clone().max_dim(1);
    let shifted = logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp;

    let mut target_one_hot = vec![0.0f32; batch_size * num_actions];
    for (i, &t) in targets.iter().enumerate() {
        target_one_hot[i * num_actions + t as usize] = 1.0;
    }
    let target_tensor = Tensor::<B, 1>::from_floats(target_one_hot.as_slice(), device)
        .reshape([batch_size, num_actions]);

    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

/// Cross-entropy loss for soft policy targets (MCTS visit distribution).
/// Invalid actions (target == 0.0) are masked out of the softmax computation
/// so that no gradient flows through their logits.
fn cross_entropy_loss_soft<B: Backend>(
    logits: Tensor<B, 2>,
    targets_flat: &[f32],
    device: &B::Device,
    num_actions: usize,
) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];

    // Build mask: -1e9 for invalid actions (target == 0), 0 for valid
    let mut mask_data = vec![0.0f32; batch_size * num_actions];
    for i in 0..batch_size {
        for a in 0..num_actions {
            if targets_flat[i * num_actions + a] == 0.0 {
                mask_data[i * num_actions + a] = -1e9;
            }
        }
    }
    let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        .reshape([batch_size, num_actions]);

    let masked_logits = logits + mask_tensor;
    let max_logits = masked_logits.clone().max_dim(1);
    let shifted = masked_logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp;

    let target_tensor =
        Tensor::<B, 1>::from_floats(targets_flat, device).reshape([batch_size, num_actions]);

    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

/// Get learning rate based on global step (AlphaZero-style stage drop).
/// Model metadata saved alongside the model file for config validation.
#[derive(Serialize, Deserialize)]
struct ModelMetadata {
    game_config: GameConfig,
    residual_channels: usize,
    num_residual_blocks: usize,
    policy_conv_channels: usize,
    value_conv_channels: usize,
    value_hidden: usize,
    film_hidden: usize,
}

fn save_model_metadata(model_path: &str, gc: &GameConfig, net_config: &PuyoNetConfig) {
    let meta = ModelMetadata {
        game_config: gc.clone(),
        residual_channels: net_config.residual_channels,
        num_residual_blocks: net_config.num_residual_blocks,
        policy_conv_channels: net_config.policy_conv_channels,
        value_conv_channels: net_config.value_conv_channels,
        value_hidden: net_config.value_hidden,
        film_hidden: net_config.film_hidden,
    };
    let path = format!("{}.config.json", model_path);
    let json = serde_json::to_string_pretty(&meta).expect("Failed to serialize metadata");
    std::fs::write(&path, json).expect("Failed to write model metadata");
    println!("Model metadata saved to {}", path);
}

fn az_lr_for_global_step(global_step: usize, stages: &[(usize, f64)]) -> f64 {
    let mut lr = stages[0].1;
    for &(threshold, stage_lr) in stages {
        if global_step >= threshold {
            lr = stage_lr;
        }
    }
    lr
}

/// Parse LR stages from "threshold:lr,threshold:lr,..." format.
/// e.g. "0:0.2,10000:0.02,30000:0.002,50000:0.0002"
fn parse_lr_stages(s: &str) -> Vec<(usize, f64)> {
    let mut stages: Vec<(usize, f64)> = s
        .split(',')
        .map(|pair| {
            let mut it = pair.split(':');
            let threshold = it
                .next()
                .expect("--lr-stages pair missing threshold")
                .trim()
                .parse::<usize>()
                .expect("--lr-stages threshold must be integer");
            let lr = it
                .next()
                .expect("--lr-stages pair missing lr")
                .trim()
                .parse::<f64>()
                .expect("--lr-stages lr must be float");
            (threshold, lr)
        })
        .collect();
    assert!(
        !stages.is_empty(),
        "--lr-stages must contain at least one stage"
    );
    stages.sort_by_key(|&(t, _)| t);
    assert!(stages[0].0 == 0, "--lr-stages first threshold must be 0");
    stages
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let alphazero_mode = args.iter().any(|a| a == "--alphazero");
    let data_dir = args
        .iter()
        .position(|a| a == "--data-dir")
        .map(|i| args[i + 1].clone());
    let global_step = args
        .iter()
        .position(|a| a == "--global-step")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--global-step requires integer")
        })
        .unwrap_or(0);
    let model_path = args
        .iter()
        .position(|a| a == "--model-path")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| MODEL_PATH.to_string());
    let artifacts_dir = args
        .iter()
        .position(|a| a == "--artifacts-dir")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "artifacts".to_string());
    let batch_size = args
        .iter()
        .position(|a| a == "--batch-size")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--batch-size requires integer")
        })
        .unwrap_or(BATCH_SIZE);
    let accum_steps = args
        .iter()
        .position(|a| a == "--accum-steps")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--accum-steps requires integer")
        })
        .unwrap_or(ACCUM_STEPS);
    let cols = args
        .iter()
        .position(|a| a == "--cols")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--cols requires integer")
        })
        .unwrap_or(3);
    let rows = args
        .iter()
        .position(|a| a == "--rows")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--rows requires integer")
        })
        .unwrap_or(8);
    let num_colors = args
        .iter()
        .position(|a| a == "--num-colors")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--num-colors requires integer")
        })
        .unwrap_or(3);

    let residual_channels = args
        .iter()
        .position(|a| a == "--residual-channels")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--residual-channels requires integer")
        })
        .unwrap_or(64);
    let num_blocks = args
        .iter()
        .position(|a| a == "--num-blocks")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--num-blocks requires integer")
        })
        .unwrap_or(6);
    let policy_conv_channels = args
        .iter()
        .position(|a| a == "--policy-conv-channels")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--policy-conv-channels requires integer")
        })
        .unwrap_or(2);
    let value_conv_channels = args
        .iter()
        .position(|a| a == "--value-conv-channels")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--value-conv-channels requires integer")
        })
        .unwrap_or(1);
    let value_hidden = args
        .iter()
        .position(|a| a == "--value-hidden")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--value-hidden requires integer")
        })
        .unwrap_or(64);
    let film_hidden = args
        .iter()
        .position(|a| a == "--film-hidden")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--film-hidden requires integer")
        })
        .unwrap_or(128);
    let lr_stages = args
        .iter()
        .position(|a| a == "--lr-stages")
        .map(|i| parse_lr_stages(&args[i + 1]))
        .unwrap_or_else(|| AZ_LR_STAGES_DEFAULT.to_vec());
    let num_steps = args
        .iter()
        .position(|a| a == "--num-steps")
        .map(|i| {
            args[i + 1]
                .parse::<usize>()
                .expect("--num-steps requires integer")
        })
        .unwrap_or(AZ_NUM_STEPS);
    let iter_lr_stages: Option<Vec<(usize, f64)>> = args
        .iter()
        .position(|a| a == "--iter-lr-stages")
        .map(|i| parse_lr_stages(&args[i + 1]));

    let data_path = args
        .iter()
        .position(|a| a == "--data-path")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "data/training_data.bin".to_string());

    std::fs::create_dir_all(&artifacts_dir).expect("Failed to create artifacts directory");

    let gc = GameConfig::new(cols, rows, num_colors);
    puyo_player::puyo_game::init_config(gc);
    let net_config = PuyoNetConfig::new()
        .with_residual_channels(residual_channels)
        .with_num_residual_blocks(num_blocks)
        .with_policy_conv_channels(policy_conv_channels)
        .with_value_conv_channels(value_conv_channels)
        .with_value_hidden(value_hidden)
        .with_film_hidden(film_hidden)
        .with_game_config(gc.clone());

    println!("Backend: CUDA (GPU)");
    println!(
        "Game config: cols={}, rows={}, num_colors={}",
        gc.cols, gc.rows, gc.num_colors
    );
    println!(
        "Net config: residual_channels={}, num_blocks={}",
        residual_channels, num_blocks
    );

    if alphazero_mode {
        println!("Mode: AlphaZero (Policy CE + Value MSE)");
        train_alphazero(
            data_dir.as_deref(),
            global_step,
            &model_path,
            &artifacts_dir,
            batch_size,
            accum_steps,
            num_steps,
            &lr_stages,
            iter_lr_stages.as_deref(),
            &gc,
            &net_config,
        );
    } else {
        println!("Mode: Supervised (Policy CE only)");
        train_supervised(&data_path, &model_path, batch_size, &gc, &net_config);
    }
}

// ---------------------------------------------------------------------------
// Supervised training (from generate-data)
// ---------------------------------------------------------------------------

fn train_supervised(
    data_path: &str,
    model_path: &str,
    batch_size: usize,
    gc: &GameConfig,
    net_config: &PuyoNetConfig,
) {
    let device: <TrainBackend as Backend>::Device = Default::default();
    // let data_path = "data/training_data.bin";

    let num_channels = gc.num_channels();
    let rows = gc.rows;
    let cols = gc.cols;
    let tensor_size = gc.tensor_size();
    let context_tensor_size = gc.context_tensor_size();
    let num_actions = gc.num_actions();

    println!("Loading dataset from {}...", data_path);
    let dataset = Dataset::load(data_path).expect("Failed to load dataset");
    let num_samples = dataset.samples.len();
    println!("Loaded {} samples", num_samples);

    let split = (num_samples as f64 * TRAIN_SPLIT_RATIO) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..];
    println!("Train: {}, Val: {}", train_samples.len(), val_samples.len());

    let mut model = net_config.init::<TrainBackend>(&device);
    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut best_val_loss = f32::MAX;
    let mut patience_counter = 0usize;

    for epoch in 0..NUM_EPOCHS {
        let lr = cosine_lr(epoch, NUM_EPOCHS);
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        shuffle_indices(&mut indices);

        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch_size = batch_end - batch_start;
            if batch_size == 0 {
                break;
            }

            let mut board_data = Vec::with_capacity(batch_size * tensor_size);
            let mut context_data = Vec::with_capacity(batch_size * context_tensor_size);
            let mut target_actions = Vec::with_capacity(batch_size);
            let mut value_targets = Vec::with_capacity(batch_size);

            for &idx in &indices[batch_start..batch_end] {
                let sample = &train_samples[idx];
                board_data.extend_from_slice(&sample.board_data);
                context_data.extend_from_slice(&sample.context_data);
                target_actions.push(sample.action_index);
                value_targets.push(sample.value_target);
            }

            let board_inputs =
                Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device).reshape([
                    batch_size,
                    num_channels,
                    rows,
                    cols,
                ]);
            let context_inputs =
                Tensor::<TrainBackend, 1>::from_floats(context_data.as_slice(), &device)
                    .reshape([batch_size, context_tensor_size]);

            let (logits, value) = model.forward(board_inputs, context_inputs);
            let policy_loss =
                cross_entropy_loss_hard(logits, &target_actions, &device, num_actions);

            let value_loss = value_mse_loss(value, &value_targets, &device);

            let loss = policy_loss + value_loss * VALUE_LOSS_WEIGHT;

            let loss_val = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("Failed to extract loss")[0];
            epoch_loss += loss_val;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = train_samples.len().div_ceil(batch_size);
                eprint!(
                    "\r  batch {}/{} loss={:.6}",
                    num_batches,
                    total_batches,
                    epoch_loss / num_batches as f32
                );
            }
        }
        eprintln!();

        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let val_loss =
            compute_val_loss_supervised(&val_model, val_samples, &val_device, batch_size, gc);

        let avg_train_loss = epoch_loss / num_batches as f32;
        println!(
            "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.6}",
            epoch + 1,
            NUM_EPOCHS,
            avg_train_loss,
            val_loss,
            lr
        );

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
            model
                .valid()
                .save_file(model_path, &BinFileRecorder::<FullPrecisionSettings>::new())
                .expect("Failed to save model");
            save_model_metadata(model_path, gc, net_config);
            println!("  -> Best model saved (val_loss={:.6})", val_loss);
        } else {
            patience_counter += 1;
            println!(
                "  -> No improvement ({}/{})",
                patience_counter, EARLY_STOPPING_PATIENCE
            );
            if patience_counter >= EARLY_STOPPING_PATIENCE {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }
    }

    println!("Training complete. Best val_loss={:.6}", best_val_loss);
}

fn compute_val_loss_supervised(
    model: &puyo_nn::model::PuyoNet<InnerBackend>,
    val_samples: &[puyo_trainer::data::Sample],
    device: &<InnerBackend as Backend>::Device,
    batch_size: usize,
    gc: &GameConfig,
) -> f32 {
    let num_channels = gc.num_channels();
    let rows = gc.rows;
    let cols = gc.cols;
    let tensor_size = gc.tensor_size();
    let context_tensor_size = gc.context_tensor_size();
    let num_actions = gc.num_actions();

    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(val_samples.len());
        let batch_size = batch_end - batch_start;
        if batch_size == 0 {
            break;
        }

        let mut board_data = Vec::with_capacity(batch_size * tensor_size);
        let mut context_data = Vec::with_capacity(batch_size * context_tensor_size);
        let mut target_actions = Vec::with_capacity(batch_size);
        let mut value_targets = Vec::with_capacity(batch_size);

        for sample in &val_samples[batch_start..batch_end] {
            board_data.extend_from_slice(&sample.board_data);
            context_data.extend_from_slice(&sample.context_data);
            target_actions.push(sample.action_index);
            value_targets.push(sample.value_target);
        }

        let board_inputs = Tensor::<InnerBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([batch_size, num_channels, rows, cols]);
        let context_inputs =
            Tensor::<InnerBackend, 1>::from_floats(context_data.as_slice(), device)
                .reshape([batch_size, context_tensor_size]);

        let (logits, value) = model.forward(board_inputs, context_inputs);
        let policy_loss = cross_entropy_loss_hard(logits, &target_actions, device, num_actions);

        let value_loss = value_mse_loss(value, &value_targets, device);

        let loss_val = policy_loss
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to extract policy loss")[0]
            + value_loss
                .into_data()
                .to_vec::<f32>()
                .expect("Failed to extract value loss")[0]
                * VALUE_LOSS_WEIGHT;
        total_loss += loss_val;
        num_batches += 1;
    }

    total_loss / num_batches.max(1) as f32
}

// ---------------------------------------------------------------------------
// AlphaZero training (from self-play data)
// ---------------------------------------------------------------------------

fn train_alphazero(
    data_dir: Option<&str>,
    global_step_start: usize,
    model_path: &str,
    artifacts_dir: &str,
    batch_size: usize,
    accum_steps: usize,
    num_steps: usize,
    lr_stages: &[(usize, f64)],
    iter_lr_stages: Option<&[(usize, f64)]>,
    gc: &GameConfig,
    net_config: &PuyoNetConfig,
) {
    let device: <TrainBackend as Backend>::Device = Default::default();

    let num_channels = gc.num_channels();
    let rows = gc.rows;
    let cols = gc.cols;
    let tensor_size = gc.tensor_size();
    let context_tensor_size = gc.context_tensor_size();
    let num_actions = gc.num_actions();

    let dataset = if let Some(dir) = data_dir {
        // Replay buffer mode: load all alphazero_iter_*.bin files from the directory
        println!("Loading AlphaZero datasets from {}...", dir);
        let mut paths: Vec<String> = std::fs::read_dir(dir)
            .expect("Failed to read data directory")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?.to_string();
                if name.starts_with("alphazero_iter_") && name.ends_with(".bin") {
                    Some(path.to_str()?.to_string())
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            panic!("No alphazero_iter_*.bin files found in {}", dir);
        }
        println!("Found {} data files", paths.len());
        AlphaZeroDataset::load_multiple(&paths).expect("Failed to load datasets")
    } else {
        // Legacy single-file mode
        let data_path = "data/alphazero_data.bin";
        println!("Loading AlphaZero dataset from {}...", data_path);
        AlphaZeroDataset::load(data_path).expect("Failed to load dataset")
    };
    let num_samples = dataset.samples.len();
    println!("Loaded {} total samples", num_samples);

    // All data used for training (no val split -- performance judged by self-play reward)
    let all_perms = puyo_trainer::data::all_color_permutations(gc.num_colors);
    let train_samples = dataset.samples;

    println!(
        "Train: {} samples (color augmented on-the-fly, no val split)",
        train_samples.len()
    );

    // Try to load existing model, otherwise init fresh
    let mut model = {
        let device_clone = device.clone();
        let path = model_path.to_string();
        let nc = net_config.clone();
        let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            nc.init::<TrainBackend>(&device_clone)
                .load_file(&path, &recorder, &device_clone)
        }));
        match load_result {
            Ok(Ok(m)) => {
                println!("Loaded existing model from {}", model_path);
                m
            }
            Ok(Err(e)) => {
                println!(
                    "Failed to load model from {}: {}. Initializing fresh",
                    model_path, e
                );
                net_config.init::<TrainBackend>(&device)
            }
            Err(_) => {
                println!(
                    "Model file {} is incompatible with current architecture. Initializing fresh",
                    model_path
                );
                net_config.init::<TrainBackend>(&device)
            }
        }
    };

    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let (effective_lr_stages, step_offset, lr_mode) = if let Some(stages) = iter_lr_stages {
        (stages, 0, "per-iter")
    } else {
        (lr_stages, global_step_start, "global")
    };
    let start_lr = az_lr_for_global_step(step_offset, effective_lr_stages);
    let end_lr = az_lr_for_global_step(
        step_offset + num_steps.saturating_sub(1),
        effective_lr_stages,
    );
    println!("AlphaZero training: {} steps, global_step={}, LR {:.0e} (end ~{:.0e}), batch_size={}, accum_steps={}, effective_batch={}, lr_mode={}",
        num_steps, global_step_start, start_lr, end_lr, batch_size, accum_steps, batch_size * accum_steps, lr_mode);

    let mut running_p_loss = 0.0f32;
    let mut running_v_loss = 0.0f32;
    let mut running_count = 0usize;
    let mut accum: GradientsAccumulator<puyo_nn::model::PuyoNet<TrainBackend>> =
        GradientsAccumulator::new();

    for step in 0..num_steps {
        let lr = az_lr_for_global_step(step_offset + step, effective_lr_stages);

        // Gradient accumulation: run accum_steps micro-batches per optimizer step
        for _micro in 0..accum_steps {
            let batch_size = batch_size.min(train_samples.len());
            let mut board_data = Vec::with_capacity(batch_size * tensor_size);
            let mut context_data = Vec::with_capacity(batch_size * context_tensor_size);
            let mut policy_targets = Vec::with_capacity(batch_size * num_actions);
            let mut value_targets = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let idx = (time_seed() >> 33) as usize % train_samples.len();
                let sample = &train_samples[idx];

                let perm_idx = (time_seed() >> 33) as usize % all_perms.len();
                let perm = &all_perms[perm_idx];

                let mut bd = sample.board_data.clone();
                let mut cd = sample.context_data.clone();
                puyo_trainer::data::apply_color_perm_board(&mut bd, perm);
                puyo_trainer::data::apply_color_perm_context(&mut cd, perm);

                board_data.extend_from_slice(&bd);
                context_data.extend_from_slice(&cd);
                policy_targets.extend_from_slice(&sample.mcts_policy);
                value_targets.push(sample.value_target);
            }

            let board_inputs =
                Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device).reshape([
                    batch_size,
                    num_channels,
                    rows,
                    cols,
                ]);
            let context_inputs =
                Tensor::<TrainBackend, 1>::from_floats(context_data.as_slice(), &device)
                    .reshape([batch_size, context_tensor_size]);

            let (logits, value) = model.forward(board_inputs, context_inputs);

            let policy_loss =
                cross_entropy_loss_soft(logits, &policy_targets, &device, num_actions);
            let value_loss = value_mse_loss(value, &value_targets, &device);

            let p_loss_val = policy_loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("Failed to extract policy loss")[0];
            let v_loss_val = value_loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("Failed to extract value loss")[0];
            // Scale loss by 1/accum_steps so accumulated gradients average correctly
            let total_loss = (policy_loss + value_loss * VALUE_LOSS_WEIGHT) / (accum_steps as f32);

            running_p_loss += p_loss_val;
            running_v_loss += v_loss_val;
            running_count += 1;

            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            accum.accumulate(&model, grads);
        }

        // Apply accumulated gradients
        let grads = accum.grads();
        model = optim.step(lr, model, grads);

        if (step + 1) % 50 == 0 {
            eprint!(
                "\r  step {}/{} p_loss={:.4} v_loss={:.4} lr={:.6}",
                step + 1,
                num_steps,
                running_p_loss / running_count as f32,
                running_v_loss / running_count as f32,
                lr,
            );
        }
    }
    eprintln!();

    let final_p = running_p_loss / running_count as f32;
    let final_v = running_v_loss / running_count as f32;

    // Save model (always -- no val-based selection, performance judged by self-play)
    model
        .valid()
        .save_file(model_path, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model");
    save_model_metadata(model_path, gc, net_config);

    let global_step_end = global_step_start + num_steps;
    println!(
        "AlphaZero training complete. final train_loss(p={:.6}, v={:.6}) global_step={}",
        final_p, final_v, global_step_end
    );

    // Write final global step to file for loop script
    let global_step_path = format!("{}/global_step.txt", artifacts_dir);
    std::fs::write(&global_step_path, global_step_end.to_string())
        .expect("Failed to write global_step.txt");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shuffle_indices(indices: &mut [usize]) {
    for i in (1..indices.len()).rev() {
        let j = (time_seed() >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}
