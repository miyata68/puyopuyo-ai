//! PvP replay training with a persistent holdout and inference-only metrics.
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use puyo_player::pvp_search::SearchRng;
use puyo_trainer::{pvp_cli::Args, pvp_metrics::*, pvp_replay::*, pvp_training::*};
fn run<B: AutodiffBackend>(args: Args) -> Result<(), String>
where
    B::Device: Default,
{
    let paths = args.data_paths()?;
    let fraction: f64 = args.number("--validation-fraction", "0.10")?;
    let validation_seed: u64 = args.number("--validation-seed", "42")?;
    let replay = ReplayPool::load(&paths, fraction, validation_seed)?;
    println!("Replay datasets:");
    for source in &replay.sources {
        println!(
            "source={} samples={} train={} val={}",
            source.path,
            source.train + source.validation,
            source.train,
            source.validation
        );
    }
    let total = replay.train().len() + replay.validation().len();
    println!("Replay total samples={total} total_train={} total_val={} validation_fraction_actual={:.6} validation_seed={validation_seed}",
        replay.train().len(), replay.validation().len(), replay.validation().len() as f64 / total as f64);
    let steps: usize = args.number("--steps", "1000")?;
    let batch: usize = args.number("--batch-size", "128")?;
    let lr: f64 = args.number("--learning-rate", "0.001")?;
    let value_weight: f32 = args.number("--value-loss-weight", "1.0")?;
    let interval: usize = args.number("--eval-interval", "100")?;
    let eval_samples: usize = args.number("--eval-samples", "4096")?;
    let eval_batch: usize = args.number("--eval-batch-size", "512")?;
    if steps == 0
        || batch == 0
        || interval == 0
        || eval_samples == 0
        || eval_batch == 0
        || !lr.is_finite()
        || lr <= 0.
        || !value_weight.is_finite()
        || value_weight < 0.
    {
        return Err("invalid training/evaluation hyperparameters".into());
    }
    let model_path = args.get("--model-path", "artifacts/pvp/puyo_pvp_model");
    let output = args.get("--output-model", "artifacts/pvp/puyo_pvp_model_next");
    let init_seed = args.number("--model-init-seed", "42")?;
    let device = B::Device::default();
    let (mut net, meta) =
        load_or_initialize::<B>(&model_path, args.net_config()?, init_seed, &device)?;
    let train_eval = fixed_subset(
        replay.train().len(),
        eval_samples,
        validation_seed ^ 0x747261696e,
    );
    let val_eval = fixed_subset(
        replay.validation().len(),
        eval_samples,
        validation_seed ^ 0x76616c,
    );
    let evaluate = |net: &_, step| -> Result<(), String> {
        println!(
            "EVAL step={step} split=train {}",
            evaluate_snapshot(net, replay.train(), &train_eval, eval_batch, &device)?
        );
        println!(
            "EVAL step={step} split=val {}",
            evaluate_snapshot(net, replay.validation(), &val_eval, eval_batch, &device)?
        );
        Ok(())
    };
    evaluate(&net, 0)?;
    let mut rng = SearchRng::new(args.number("--seed", "42")?);
    let perms = puyo_trainer::data::all_color_permutations(4);
    let mut optimizer = AdamConfig::new().init();
    for step in 1..=steps {
        let samples = replay.training_batch(batch, &mut rng, &perms)?;
        let (logits, value) = forward_samples(&net, &samples, &device);
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
            "TRAIN step={step} policy_ce={p:.6} value_mse={v:.6} total={:.6}",
            p + value_weight * v
        );
        if step % interval == 0 || step == steps {
            evaluate(&net, step)?;
        }
    }
    let all_val: Vec<_> = (0..replay.validation().len()).collect();
    println!(
        "FINAL_VALIDATION {}",
        evaluate_snapshot(&net, replay.validation(), &all_val, eval_batch, &device)?
    );
    save_model(net.valid(), &meta, &output)?;
    println!("trained_steps={steps} train_samples={} validation_samples={} value_loss_weight={value_weight} model={output}", replay.train().len(), replay.validation().len());
    Ok(())
}
fn main() {
    let result = (|| {
        let args = Args::parse(&[
            "--data",
            "--data-paths",
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
            "--validation-fraction",
            "--validation-seed",
            "--eval-interval",
            "--eval-samples",
            "--eval-batch-size",
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
