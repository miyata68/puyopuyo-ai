//! Separate PvP CE + win/loss MSE trainer. No solo value transformation.
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use puyo_core::pvp_encoding::*;
use puyo_player::pvp_search::SearchRng;
use puyo_trainer::{pvp_cli::Args, pvp_data::*, pvp_training::*};
fn run<B: AutodiffBackend>(args: Args) -> Result<(), String>
where
    B::Device: Default,
{
    let input = args.get("--data", "data/pvp/pvp_alphazero_iter_001.bin");
    let dataset = PvpAlphaZeroDataset::load(&input).map_err(|e| e.to_string())?;
    if dataset.samples.is_empty() {
        return Err("PvP dataset is empty (all games may be truncated)".into());
    }
    let steps: usize = args.number("--steps", "1000")?;
    let batch: usize = args.number("--batch-size", "128")?;
    let lr: f64 = args.number("--learning-rate", "0.001")?;
    let value_weight: f32 = args.number("--value-loss-weight", "1.0")?;
    if steps == 0
        || batch == 0
        || !lr.is_finite()
        || lr <= 0.
        || !value_weight.is_finite()
        || value_weight < 0.
    {
        return Err("invalid training hyperparameters".into());
    }
    let model_path = args.get("--model-path", "artifacts/pvp/puyo_pvp_model");
    let output = args.get("--output-model", "artifacts/pvp/puyo_pvp_model_next");
    let init_seed = args.number("--model-init-seed", "42")?;
    let device = B::Device::default();
    let (mut net, meta) =
        load_or_initialize::<B>(&model_path, args.net_config()?, init_seed, &device)?;
    let mut rng = SearchRng::new(args.number("--seed", "42")?);
    let perms = puyo_trainer::data::all_color_permutations(4);
    let mut optimizer = AdamConfig::new().init();
    for step in 0..steps {
        let mut samples = Vec::with_capacity(batch);
        for _ in 0..batch {
            let mut sample =
                dataset.samples[rng.next_u64() as usize % dataset.samples.len()].clone();
            augment_colors(&mut sample, &perms[rng.next_u64() as usize % perms.len()])?;
            samples.push(sample);
        }
        let board: Vec<_> = samples
            .iter()
            .flat_map(|s| s.board_data.iter().copied())
            .collect();
        let context: Vec<_> = samples
            .iter()
            .flat_map(|s| s.context_data.iter().copied())
            .collect();
        let (logits, value) = net.forward(
            Tensor::<B, 1>::from_floats(board.as_slice(), &device).reshape([
                batch,
                BOARD_CHANNELS,
                ROWS,
                COLS,
            ]),
            Tensor::<B, 1>::from_floats(context.as_slice(), &device).reshape([batch, CONTEXT_SIZE]),
        );
        let (p_loss, v_loss) = losses(logits, value, &samples, &device);
        let p = p_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?[0];
        let v = v_loss
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| format!("{e:?}"))?[0];
        if !p.is_finite() || !v.is_finite() {
            return Err("non-finite PvP training loss".into());
        }
        let grads = (p_loss + v_loss * value_weight).backward();
        let grads = GradientsParams::from_grads(grads, &net);
        net = optimizer.step(lr, net, grads);
        println!(
            "step={} policy_ce={p:.6} value_mse={v:.6} total={:.6}",
            step + 1,
            p + value_weight * v
        );
    }
    save_model(net.valid(), &meta, &output)?;
    println!(
        "trained_steps={steps} samples={} value_loss_weight={value_weight} model={output}",
        dataset.samples.len()
    );
    Ok(())
}
fn main() {
    let result = (|| {
        let args = Args::parse(&[
            "--data",
            "--steps",
            "--batch-size",
            "--learning-rate",
            "--value-loss-weight",
            "--model-path",
            "--output-model",
            "--model-init-seed",
            "--seed",
            "--backend",
            "--residual-channels",
            "--residual-blocks",
        ])?;
        match args.get("--backend", "cpu").as_str() {
            "cpu" => run::<burn::backend::Autodiff<burn::backend::NdArray>>(args),
            #[cfg(feature = "gpu")]
            "cuda" => run::<burn::backend::Autodiff<burn::backend::Cuda<f32>>>(args),
            _ => Err("unsupported backend (use cpu, or cuda with gpu feature)".into()),
        }
    })();
    if let Err(e) = result {
        eprintln!("train-pvp: {e}");
        std::process::exit(1);
    }
}
