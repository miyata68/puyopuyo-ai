//! AlphaZero-style self-play for Puyo Puyo (Gumbel MCTS).
//!
//! Plays games using Gumbel MCTS + neural network, collects training data,
//! and saves it for the training binary.
//!
//! A dedicated GPU thread batches inference requests from game threads.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings};

use az_framework::game::Game;
use puyo_core::config::GameConfig;
use puyo_core::game::{GamePhase, GameState};
use puyo_core::rand::time_seed;
use puyo_core::state::PuyoState;
use puyo_nn::model::{PuyoNet, PuyoNetConfig};
use puyo_player::mcts::{mcts_search, mcts_search_batched, InferenceProvider};
use puyo_player::nn_eval::MctsConfig;
use puyo_player::puyo_game::PuyoGame;
use puyo_trainer::data::{AlphaZeroDataset, AlphaZeroSample};

use burn::backend::Cuda;
use puyo_player::inference_server;
use puyo_player::nn_eval::PuyoGameModel;

const MODEL_PATH: &str = "artifacts/puyo_model";
const DEFAULT_OUTPUT_PATH: &str = "data/alphazero_data.bin";
const MAX_TURNS: u32 = 80;
const DEFAULT_GPU_THREADS: usize = 128;
const DEFAULT_MAX_BATCH_SIZE: usize = 128;

struct Args {
    num_games: u64,
    num_simulations: usize,
    c_puct_init: f32,
    c_puct_base: f32,
    m: usize,
    c_visit: f32,
    gamma: f32,
    output_path: String,
    model_path: String,
    threads: Option<usize>,
    batch_size: Option<usize>,
    num_leaves: usize,
    min_chain: u32,
    cols: usize,
    rows: usize,
    num_colors: usize,
    residual_channels: usize,
    num_blocks: usize,
    policy_conv_channels: usize,
    value_conv_channels: usize,
    value_hidden: usize,
    film_hidden: usize,
    fp16: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        num_games: 300,
        num_simulations: 64,
        c_puct_init: 1.5,
        c_puct_base: 19652.0,
        m: 16,
        c_visit: 5.0,
        gamma: 0.95,
        output_path: DEFAULT_OUTPUT_PATH.to_string(),
        model_path: MODEL_PATH.to_string(),
        threads: None,
        batch_size: None,
        num_leaves: 1,
        min_chain: 0,
        cols: 3,
        rows: 8,
        num_colors: 3,
        residual_channels: 64,
        num_blocks: 6,
        policy_conv_channels: 2,
        value_conv_channels: 1,
        value_hidden: 64,
        film_hidden: 128,
        fp16: false,
    };
    let next_val = |i: usize, flag: &str| -> &String {
        args.get(i).unwrap_or_else(|| {
            eprintln!("Error: {flag} requires a value");
            std::process::exit(1);
        })
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                result.num_games = next_val(i, "--games")
                    .parse()
                    .expect("--games requires integer");
            }
            "--simulations" => {
                i += 1;
                result.num_simulations = next_val(i, "--simulations")
                    .parse()
                    .expect("--simulations requires integer");
            }
            "--c-puct-init" => {
                i += 1;
                result.c_puct_init = next_val(i, "--c-puct-init")
                    .parse()
                    .expect("--c-puct-init requires float");
            }
            "--c-puct-base" => {
                i += 1;
                result.c_puct_base = next_val(i, "--c-puct-base")
                    .parse()
                    .expect("--c-puct-base requires float");
            }
            "--m" => {
                i += 1;
                result.m = next_val(i, "--m").parse().expect("--m requires integer");
            }
            "--c-visit" => {
                i += 1;
                result.c_visit = next_val(i, "--c-visit")
                    .parse()
                    .expect("--c-visit requires float");
            }
            "--gamma" => {
                i += 1;
                result.gamma = next_val(i, "--gamma")
                    .parse()
                    .expect("--gamma requires float");
            }
            "--output" => {
                i += 1;
                result.output_path = next_val(i, "--output").clone();
            }
            "--model-path" => {
                i += 1;
                result.model_path = next_val(i, "--model-path").clone();
            }
            "--threads" => {
                i += 1;
                result.threads = Some(
                    next_val(i, "--threads")
                        .parse()
                        .expect("--threads requires integer"),
                );
            }
            "--batch-size" => {
                i += 1;
                result.batch_size = Some(
                    next_val(i, "--batch-size")
                        .parse()
                        .expect("--batch-size requires integer"),
                );
            }
            "--num-leaves" => {
                i += 1;
                result.num_leaves = next_val(i, "--num-leaves")
                    .parse()
                    .expect("--num-leaves requires integer");
            }
            "--min-chain" => {
                i += 1;
                result.min_chain = next_val(i, "--min-chain")
                    .parse()
                    .expect("--min-chain requires integer");
            }
            "--cols" => {
                i += 1;
                result.cols = next_val(i, "--cols")
                    .parse()
                    .expect("--cols requires integer");
            }
            "--rows" => {
                i += 1;
                result.rows = next_val(i, "--rows")
                    .parse()
                    .expect("--rows requires integer");
            }
            "--num-colors" => {
                i += 1;
                result.num_colors = next_val(i, "--num-colors")
                    .parse()
                    .expect("--num-colors requires integer");
            }
            "--residual-channels" => {
                i += 1;
                result.residual_channels = next_val(i, "--residual-channels")
                    .parse()
                    .expect("--residual-channels requires integer");
            }
            "--num-blocks" => {
                i += 1;
                result.num_blocks = next_val(i, "--num-blocks")
                    .parse()
                    .expect("--num-blocks requires integer");
            }
            "--policy-conv-channels" => {
                i += 1;
                result.policy_conv_channels = next_val(i, "--policy-conv-channels")
                    .parse()
                    .expect("--policy-conv-channels requires integer");
            }
            "--value-conv-channels" => {
                i += 1;
                result.value_conv_channels = next_val(i, "--value-conv-channels")
                    .parse()
                    .expect("--value-conv-channels requires integer");
            }
            "--value-hidden" => {
                i += 1;
                result.value_hidden = next_val(i, "--value-hidden")
                    .parse()
                    .expect("--value-hidden requires integer");
            }
            "--film-hidden" => {
                i += 1;
                result.film_hidden = next_val(i, "--film-hidden")
                    .parse()
                    .expect("--film-hidden requires integer");
            }
            "--fp16" => {
                result.fp16 = true;
            }
            other => eprintln!("Unknown option: {} (ignoring)", other),
        }
        i += 1;
    }
    result
}

/// Record from a single move during self-play.
struct MoveRecord {
    board_data: Vec<f32>,
    context_data: Vec<f32>,
    mcts_policy: Vec<f32>,
    reward: f32,
}

/// Result from a single self-play game.
struct GameResult {
    samples: Vec<AlphaZeroSample>,
    max_chain: u32,
    total_reward: f32,
}

/// Play one self-play game and return training samples.
fn play_one_game(provider: &dyn InferenceProvider, args: &Args, gc: &GameConfig) -> GameResult {
    let cols = gc.cols;
    let mut game = GameState::new(gc.clone());
    let mut move_records: Vec<MoveRecord> = Vec::with_capacity(MAX_TURNS as usize);
    let mut move_count = 0u32;

    let mcts_config = MctsConfig {
        num_simulations: args.num_simulations,
        c_puct_init: args.c_puct_init,
        c_puct_base: args.c_puct_base,
        m: args.m,
        c_visit: args.c_visit,
        gamma: args.gamma,
        num_leaves: args.num_leaves,
        ..Default::default()
    };

    while game.phase != GamePhase::GameOver && move_count < MAX_TURNS {
        let current_piece = game.current_piece.as_ref().unwrap().piece;

        let puyo_state = PuyoState {
            board: game.board.clone(),
            current: current_piece,
            next: game.next_piece,
            next_next: game.next_next_piece,
        };

        let board_data = PuyoGame::encode_board(&puyo_state);
        let context_data = PuyoGame::encode_context(&puyo_state);

        let (mcts_policy, _q_values) = if mcts_config.num_leaves > 1 {
            mcts_search_batched::<PuyoGame>(&puyo_state, provider, &mcts_config)
        } else {
            mcts_search::<PuyoGame>(&puyo_state, provider, &mcts_config)
        };

        let valid_mask = PuyoGame::valid_action_mask(&puyo_state);
        let action = select_from_policy(&mcts_policy, &valid_mask);

        let placement = puyo_core::placement::index_to_placement(action, cols);
        let chain_result = game.apply_placement(&placement);

        move_records.push(MoveRecord {
            board_data,
            context_data,
            mcts_policy,
            reward: chain_result.score as f32,
        });

        move_count += 1;
    }

    let num_moves = move_records.len();
    let total_reward: f32 = move_records.iter().map(|r| r.reward).sum();
    let truncated = move_count >= MAX_TURNS && game.phase != GamePhase::GameOver;
    let mut samples = Vec::new();
    if num_moves > 0 {
        let bootstrap_value = if truncated {
            estimate_value(provider, &game)
        } else {
            0.0
        };
        let mut value_targets = vec![0.0f32; num_moves];
        value_targets[num_moves - 1] =
            move_records[num_moves - 1].reward + args.gamma * bootstrap_value;
        for i in (0..num_moves - 1).rev() {
            value_targets[i] = move_records[i].reward + args.gamma * value_targets[i + 1];
        }

        for (i, record) in move_records.into_iter().enumerate() {
            samples.push(AlphaZeroSample {
                board_data: record.board_data,
                context_data: record.context_data,
                mcts_policy: record.mcts_policy,
                value_target: value_targets[i],
            });
        }
    }

    GameResult {
        samples,
        max_chain: game.max_chain,
        total_reward,
    }
}

fn validate_model_metadata(model_path: &str, gc: &GameConfig) {
    let meta_path = format!("{}.config.json", model_path);
    if let Ok(json) = std::fs::read_to_string(&meta_path) {
        #[derive(serde::Deserialize)]
        struct ModelMetadata {
            game_config: GameConfig,
        }
        if let Ok(meta) = serde_json::from_str::<ModelMetadata>(&json) {
            if meta.game_config != *gc {
                eprintln!(
                    "WARNING: Model was trained with {:?} but current config is {:?}",
                    meta.game_config, gc
                );
            }
        }
    }
}

fn load_model<B: Backend, S: burn::record::PrecisionSettings>(
    device: &B::Device,
    model_path: &str,
    net_config: &PuyoNetConfig,
) -> PuyoNet<B> {
    let config = net_config.clone();
    let path = model_path.to_string();
    let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let recorder = BinFileRecorder::<S>::new();
        config.init::<B>(device).load_file(&path, &recorder, device)
    }));
    match load_result {
        Ok(Ok(m)) => {
            println!("Loaded existing model from {}", model_path);
            m
        }
        Ok(Err(e)) => {
            println!(
                "Failed to load model from {}: {}. Initializing random weights",
                model_path, e
            );
            config.init::<B>(device)
        }
        Err(_) => {
            println!(
                "Model file {} is incompatible with current architecture. Initializing random weights",
                model_path
            );
            config.init::<B>(device)
        }
    }
}

fn run_with_backend<B: Backend + 'static>(
    fp16: bool,
    args: &Args,
    gc: &GameConfig,
    net_config: &PuyoNetConfig,
) {
    let precision_label = if fp16 { "f16" } else { "f32" };
    println!(
        "Backend: Cuda (GPU, {}) — Gumbel MCTS self-play (batched)",
        precision_label
    );
    let device: B::Device = Default::default();

    let model = if fp16 {
        load_model::<B, HalfPrecisionSettings>(&device, &args.model_path, net_config)
    } else {
        load_model::<B, FullPrecisionSettings>(&device, &args.model_path, net_config)
    };

    let num_threads = args.threads.unwrap_or(DEFAULT_GPU_THREADS);
    let batch_size = args.batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE);
    println!(
        "Using {} game threads, max_batch_size={}",
        num_threads, batch_size
    );

    let client = inference_server::start_inference_server(
        PuyoGameModel::with_config(model, gc.clone()),
        device,
        batch_size,
    );

    run_games_parallel(args, gc, num_threads, |_| {
        let thread_client = client.clone();
        (thread_client, 0usize)
    });

    // Skip destructors to avoid CUDA cleanup crash (double free on exit)
    std::process::exit(0);
}

fn main() {
    let args = parse_args();
    let gc = GameConfig::new(args.cols, args.rows, args.num_colors);
    puyo_player::puyo_game::init_config(gc);
    let net_config = PuyoNetConfig::new()
        .with_residual_channels(args.residual_channels)
        .with_num_residual_blocks(args.num_blocks)
        .with_policy_conv_channels(args.policy_conv_channels)
        .with_value_conv_channels(args.value_conv_channels)
        .with_value_hidden(args.value_hidden)
        .with_film_hidden(args.film_hidden)
        .with_game_config(gc.clone());
    validate_model_metadata(&args.model_path, &gc);

    println!(
        "games={}, simulations={}, c_puct_init={}, c_puct_base={}, m={}, c_visit={}, gamma={}, num_leaves={}, min_chain={}",
        args.num_games, args.num_simulations, args.c_puct_init, args.c_puct_base, args.m,
        args.c_visit, args.gamma, args.num_leaves, args.min_chain,
    );
    println!(
        "Game config: cols={}, rows={}, num_colors={}, residual_channels={}, num_blocks={}, policy_conv_ch={}, value_conv_ch={}, value_hidden={}, film_hidden={}",
        gc.cols, gc.rows, gc.num_colors, args.residual_channels, args.num_blocks,
        args.policy_conv_channels, args.value_conv_channels, args.value_hidden, args.film_hidden,
    );

    if args.fp16 {
        run_with_backend::<Cuda<half::f16>>(true, &args, &gc, &net_config);
    } else {
        run_with_backend::<Cuda<f32>>(false, &args, &gc, &net_config);
    }
}

/// Run self-play games in parallel.
/// `make_provider_data` is called once per thread from the main thread,
/// returning a tuple of (provider, extra_data). The provider must be Send.
fn run_games_parallel<F, P>(args: &Args, gc: &GameConfig, num_threads: usize, make_provider_data: F)
where
    F: Fn(usize) -> (P, usize),
    P: InferenceProvider + Send,
{
    let start_time = std::time::Instant::now();

    let games_done = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let total_max_chain = AtomicU32::new(0);
    let total_chain_sum = AtomicU64::new(0);
    let total_reward_sum = AtomicU64::new(0);

    let games_per_thread = args.num_games.div_ceil(num_threads as u64);

    // Pre-create providers on main thread
    let providers: Vec<P> = (0..num_threads).map(|i| make_provider_data(i).0).collect();

    let game_results: Vec<GameResult> = std::thread::scope(|s| {
        let handles: Vec<_> = providers
            .into_iter()
            .enumerate()
            .map(|(thread_idx, thread_provider)| {
                let start = thread_idx as u64 * games_per_thread;
                let end = (start + games_per_thread).min(args.num_games);

                let args = &args;
                let games_done = &games_done;
                let total_samples = &total_samples;
                let total_max_chain = &total_max_chain;
                let total_chain_sum = &total_chain_sum;
                let total_reward_sum = &total_reward_sum;
                let start_time = &start_time;

                s.spawn(move || {
                    let mut thread_results = Vec::new();
                    for _ in start..end {
                        let result = play_one_game(&thread_provider, args, gc);

                        let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                        total_samples.fetch_add(result.samples.len() as u64, Ordering::Relaxed);
                        total_max_chain.fetch_max(result.max_chain, Ordering::Relaxed);
                        total_chain_sum.fetch_add(result.max_chain as u64, Ordering::Relaxed);
                        total_reward_sum.fetch_add(result.total_reward as u64, Ordering::Relaxed);

                        let elapsed = start_time.elapsed().as_secs_f64();
                        let games_per_sec = done as f64 / elapsed;
                        let samples_so_far = total_samples.load(Ordering::Relaxed);
                        let max_chain = total_max_chain.load(Ordering::Relaxed);
                        let chain_sum = total_chain_sum.load(Ordering::Relaxed);
                        let avg_chain = chain_sum as f64 / done as f64;
                        let reward_sum = total_reward_sum.load(Ordering::Relaxed);
                        let avg_reward = reward_sum as f64 / done as f64;
                        let eta = (args.num_games - done) as f64 / games_per_sec;
                        println!(
                            "[{:>4}/{}] samples: {:>6} | chain(game/max/avg): {}/{}/{:.1} | reward(game/avg): {}/{:.0} | {:.2} games/s | ETA: {:.0}s",
                            done, args.num_games, samples_so_far,
                            result.max_chain, max_chain, avg_chain,
                            result.total_reward as u64, avg_reward,
                            games_per_sec, eta,
                        );

                        thread_results.push(result);
                    }
                    thread_results
                })
            })
            .collect();

        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    let mut dataset = AlphaZeroDataset::new();
    let mut final_max_chain = 0u32;
    let mut filtered_games = 0u32;
    let total_games = game_results.len() as u32;
    for result in game_results {
        final_max_chain = final_max_chain.max(result.max_chain);
        if result.max_chain >= args.min_chain {
            dataset.samples.extend(result.samples);
            filtered_games += 1;
        }
    }

    if args.min_chain > 0 {
        println!(
            "Chain filter: {}/{} games passed (min_chain={})",
            filtered_games, total_games, args.min_chain
        );
    }

    println!(
        "Self-play complete: {} games, {} samples, max chain: {}",
        args.num_games,
        dataset.samples.len(),
        final_max_chain
    );

    std::fs::create_dir_all("data").expect("Failed to create data directory");
    dataset
        .save(&args.output_path)
        .expect("Failed to save dataset");
    println!("Saved to {}", args.output_path);
}

/// Estimate the value of the current game state using the neural network.
fn estimate_value(provider: &dyn InferenceProvider, game: &GameState) -> f32 {
    let current_piece = match &game.current_piece {
        Some(fp) => fp.piece,
        None => return 0.0,
    };
    let puyo_state = PuyoState {
        board: game.board.clone(),
        current: current_piece,
        next: game.next_piece,
        next_next: game.next_next_piece,
    };
    let board_data = PuyoGame::encode_board(&puyo_state);
    let context_data = PuyoGame::encode_context(&puyo_state);

    let (_logits, value) = provider.infer(&board_data, &context_data);
    value
}

/// Select an action by sampling from the MCTS policy (valid actions only).
fn select_from_policy(policy: &[f32], valid_mask: &[bool]) -> usize {
    let r = (time_seed() as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cumulative += p as f64;
        if r < cumulative {
            return i;
        }
    }
    // Fallback: pick the valid action with the highest policy value
    policy
        .iter()
        .enumerate()
        .filter(|(i, _)| valid_mask.get(*i).copied().unwrap_or(false))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
