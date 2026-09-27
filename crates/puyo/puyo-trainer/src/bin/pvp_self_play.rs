//! PvP self-play. CPU by default for deterministic smoke tests; --backend cuda
//! reuses the existing inference server and can batch across game threads.
use az_framework::inference_server::start_inference_server;
use burn::prelude::*;
use puyo_core::pvp::MatchResult;
use puyo_player::pvp_model::PvpGameModel;
use puyo_player::pvp_search::PvpSearchConfig;
use puyo_trainer::{
    pvp_cli::Args, pvp_data::PvpAlphaZeroDataset, pvp_self_play::*, pvp_training::*,
};
use rayon::prelude::*;
fn run<B: Backend + 'static>(args: Args) -> Result<(), String>
where
    B::Device: Default + Send,
{
    let games: usize = args.number("--games", "100")?;
    let threads: usize = args.number("--threads", "1")?;
    let batch: usize = args.number("--inference-batch-size", "32")?;
    if games == 0 || threads == 0 || batch == 0 {
        return Err("games/threads/batch must be positive".into());
    }
    let seed: u64 = args.number("--seed-offset", "100000")?;
    let init_seed: u64 = args.number("--model-init-seed", "42")?;
    let model_path = args.get("--model-path", "artifacts/pvp/puyo_pvp_model");
    let output = args.get("--output", "data/pvp/pvp_alphazero_iter_001.bin");
    let noise: bool = args.number("--root-noise", "true")?;
    let swap: bool = args.number("--swap-streams", "false")?;
    let replay: bool = args.number("--verify-replay", "false")?;
    let config = SelfPlayConfig {
        search: PvpSearchConfig {
            simulations: args.number("--simulations", "64")?,
            root_noise: noise,
            ..Default::default()
        },
        max_ticks: args.number("--max-ticks", "2000")?,
        temperature: args.number("--temperature", "1.0")?,
        temperature_drop_tick: args.number("--temperature-drop-tick", "100")?,
        streams: if swap { [1, 0] } else { [0, 1] },
    };
    config.search.validate()?;
    let device = B::Device::default();
    let (net, meta) = load_or_initialize::<B>(&model_path, args.net_config()?, init_seed, &device)?;
    if !std::path::Path::new(&format!("{model_path}.bin")).exists() {
        save_model(net.clone(), &meta, &model_path)?;
    }
    println!("root_noise={} temperature={} temperature_drop_tick={} streams={:?} simulations={} shared_model=true",noise,config.temperature,config.temperature_drop_tick,config.streams,config.search.simulations);
    let client = start_inference_server::<B, _>(PvpGameModel { net }, device, batch);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;
    // Indexed collection preserves seed order independently of worker scheduling.
    let results: Vec<_> = pool.install(|| {
        (0..games)
            .into_par_iter()
            .map(|i| {
                let client = client.clone();
                let game = play_match(&client, seed.wrapping_add(i as u64), &config)?;
                if replay {
                    let repeated = play_match(&client, seed.wrapping_add(i as u64), &config)?;
                    if game != repeated {
                        return Err(format!(
                            "replay mismatch seed={}",
                            seed.wrapping_add(i as u64)
                        ));
                    }
                }
                Ok::<_, String>(game)
            })
            .collect()
    });
    let mut dataset = PvpAlphaZeroDataset::default();
    let mut wins = [0usize; 4];
    let mut samples = [0usize; 2];
    let mut ticks = 0;
    let mut chains = [0u64; 2];
    let mut max_chain = [0u32; 2];
    let mut sent = 0;
    let mut offset = 0;
    let mut dropped = 0;
    let mut identical = 0;
    let mut simultaneous = 0;
    let mut breaks = Vec::new();
    let mut long_symmetric = 0;
    for (i, result) in results.into_iter().enumerate() {
        let game = result?;
        let m = game.metrics;
        wins[match m.result {
            MatchResult::Player1Win => 0,
            MatchResult::Player2Win => 1,
            MatchResult::Draw => 2,
            MatchResult::Ongoing => 3,
        }] += 1;
        ticks += m.ticks;
        for id in 0..2 {
            samples[id] += m.samples[id];
            chains[id] += m.max_chain[id] as u64;
            max_chain[id] = max_chain[id].max(m.max_chain[id]);
        }
        sent += m.garbage_sent;
        offset += m.garbage_offset;
        dropped += m.garbage_dropped;
        identical += m.identical_action_ticks;
        simultaneous += m.simultaneous_action_ticks;
        if let Some(t) = m.symmetry_break_tick {
            breaks.push(t);
        }
        if m.ticks > 50 && m.symmetry_break_tick.is_none_or(|t| t > 50) {
            long_symmetric += 1;
        }
        println!("game={} seed={} result={:?} truncated={} ticks={} samples={:?} max_chain={:?} identical_action_rate={:.4} symmetry_break_tick={:?}",i,seed.wrapping_add(i as u64),m.result,m.truncated,m.ticks,m.samples,m.max_chain,m.identical_action_rate(),m.symmetry_break_tick);
        dataset.samples.extend(game.samples);
    }
    create_parent(&output)?;
    dataset.save(&output).map_err(|e| e.to_string())?;
    let rate = if simultaneous == 0 {
        0.
    } else {
        identical as f64 / simultaneous as f64
    };
    let avg_break = if breaks.is_empty() {
        None
    } else {
        Some(breaks.iter().sum::<u64>() as f64 / breaks.len() as f64)
    };
    println!("SUMMARY games={games} completed={} p1={} p2={} draws={} truncated={} truncation_rate={:.4} avg_ticks={:.2} samples_p1={} samples_p2={} samples={} avg_max_chain_p1={:.2} avg_max_chain_p2={:.2} max_chain={max_chain:?} garbage_sent={sent} offset={offset} dropped={dropped} identical_action_ticks={identical} simultaneous_action_ticks={simultaneous} identical_action_rate={rate:.4} asymmetric_games={} average_symmetry_break_tick={avg_break:?} replay_verified={replay}",games-wins[3],wins[0],wins[1],wins[2],wins[3],wins[3] as f64/games as f64,ticks as f64/games as f64,samples[0],samples[1],dataset.samples.len(),chains[0] as f64/games as f64,chains[1] as f64/games as f64,breaks.len());
    if wins[3] * 100 > games {
        eprintln!("WARNING: more than 1% truncated; their samples were discarded");
    }
    if long_symmetric * 100 >= games * 95 {
        eprintln!("WARNING: at least 95% of games remained symmetric for >50 ticks");
    }
    println!("dataset={output}");
    Ok(())
}
fn main() {
    let result = (|| {
        let args = Args::parse(&[
            "--games",
            "--simulations",
            "--model-path",
            "--output",
            "--seed-offset",
            "--max-ticks",
            "--temperature",
            "--temperature-drop-tick",
            "--root-noise",
            "--model-init-seed",
            "--backend",
            "--threads",
            "--inference-batch-size",
            "--residual-channels",
            "--residual-blocks",
            "--swap-streams",
            "--verify-replay",
        ])?;
        match args.get("--backend", "cpu").as_str() {
            "cpu" => run::<burn::backend::NdArray>(args),
            #[cfg(feature = "gpu")]
            "cuda" => run::<burn::backend::Cuda<f32>>(args),
            _ => Err("unsupported backend (use cpu, or cuda with gpu feature)".into()),
        }
    })();
    if let Err(e) = result {
        eprintln!("pvp-self-play: {e}");
        std::process::exit(1);
    }
}
