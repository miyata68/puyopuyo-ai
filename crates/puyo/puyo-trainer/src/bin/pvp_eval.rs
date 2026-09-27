//! Same-seed seat-swapped matches between independently loaded PvP models.
use az_framework::inference_server::start_inference_server;
use burn::prelude::*;
use puyo_player::pvp_model::PvpGameModel;
use puyo_trainer::{pvp_cli::Args, pvp_evaluation::*, pvp_training::*};
use rayon::prelude::*;
fn run<B: Backend + 'static>(args: Args) -> Result<(), String>
where
    B::Device: Default + Send,
{
    let path_a = args.get("--model-a", "");
    let path_b = args.get("--model-b", "");
    // Evaluations must never silently initialize an absent checkpoint.
    let meta_a = PvpModelMetadata::read(&path_a)?;
    let meta_b = PvpModelMetadata::read(&path_b)?;
    if meta_a.dimensions != meta_b.dimensions {
        return Err("incompatible model dimensions".into());
    }
    let pairs: usize = args.number("--pairs", "250")?;
    let threads: usize = args.number("--threads", "1")?;
    let batch: usize = args.number("--inference-batch-size", "32")?;
    let seed: u64 = args.number("--seed-offset", "2000000")?;
    if pairs == 0 || threads == 0 || batch == 0 {
        return Err("pairs/threads/batch must be positive".into());
    }
    let config = EvalConfig {
        simulations: args.number("--simulations", "64")?,
        max_ticks: args.number("--max-ticks", "2000")?,
    };
    config.search_config().validate()?;
    let device = B::Device::default();
    let (net_a, _) = load_or_initialize::<B>(&path_a, meta_a.architecture, 0, &device)?;
    let (net_b, _) = load_or_initialize::<B>(&path_b, meta_b.architecture, 0, &device)?;
    println!("model_a={path_a} model_b={path_b} pairs={pairs} simulations={} seed_offset={seed} root_noise=false temperature=0 threads={threads} inference_batch_size={batch}", config.simulations);
    let a = start_inference_server::<B, _>(PvpGameModel { net: net_a }, device.clone(), batch);
    let b = start_inference_server::<B, _>(PvpGameModel { net: net_b }, device, batch);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;
    let results: Vec<_> = pool.install(|| {
        (0..pairs)
            .into_par_iter()
            .map(|i| evaluate_pair(&a.clone(), &b.clone(), seed.wrapping_add(i as u64), &config))
            .collect()
    });
    let mut summary = EvalSummary::default();
    for (i, pair) in results.into_iter().enumerate() {
        let pair = pair?;
        for game in &pair.games {
            println!(
                "pair={i} seed={} A_seat={:?} result={:?} truncated={} ticks={}",
                game.seed,
                game.model_a_seat,
                game.result,
                game.outcome_a().is_none(),
                game.ticks
            );
        }
        summary.add_pair(&pair);
    }
    summary.print();
    Ok(())
}
fn main() {
    let result = (|| {
        let args = Args::parse(&[
            "--model-a",
            "--model-b",
            "--pairs",
            "--simulations",
            "--seed-offset",
            "--max-ticks",
            "--backend",
            "--threads",
            "--inference-batch-size",
        ])?;
        match args.get("--backend", "cpu").as_str() {
            "cpu" => run::<burn::backend::NdArray>(args),
            #[cfg(feature = "gpu")]
            "cuda" => run::<burn::backend::Cuda<f32>>(args),
            _ => Err("unsupported backend (use cpu, or cuda with gpu feature)".into()),
        }
    })();
    if let Err(e) = result {
        eprintln!("pvp-eval: {e}");
        std::process::exit(1);
    }
}
