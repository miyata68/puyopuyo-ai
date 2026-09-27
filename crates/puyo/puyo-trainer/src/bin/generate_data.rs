use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use az_framework::game::Game;

use puyo_core::config::GameConfig;
use puyo_core::game::{GamePhase, GameState};
use puyo_core::state::PuyoState;

use puyo_player::eval::SimulationEvaluator;
use puyo_player::placement::placement_to_index;
use puyo_player::puyo_game::PuyoGame;
use puyo_player::Evaluator;

use puyo_trainer::data::{Dataset, Sample};

const DEFAULT_NUM_GAMES: u64 = 10_000;
const DEFAULT_OUTPUT_PATH: &str = "data/training_data.bin";
const DEFAULT_MAX_MOVES: usize = 50;

// 従来環境
const DEFAULT_COLS: usize = 3;
const DEFAULT_ROWS: usize = 8;
const DEFAULT_NUM_COLORS: usize = 3;

struct Args {
    num_games: u64,
    output_path: String,
    max_moves: usize,

    cols: usize,
    rows: usize,
    num_colors: usize,

    threads: Option<usize>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut result = Args {
        num_games: DEFAULT_NUM_GAMES,
        output_path: DEFAULT_OUTPUT_PATH.to_string(),
        max_moves: DEFAULT_MAX_MOVES,

        cols: DEFAULT_COLS,
        rows: DEFAULT_ROWS,
        num_colors: DEFAULT_NUM_COLORS,

        threads: None,
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

            "--output" => {
                i += 1;
                result.output_path = next_val(i, "--output").clone();
            }

            "--max-moves" => {
                i += 1;
                result.max_moves = next_val(i, "--max-moves")
                    .parse()
                    .expect("--max-moves requires integer");
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

            "--threads" => {
                i += 1;
                result.threads = Some(
                    next_val(i, "--threads")
                        .parse()
                        .expect("--threads requires integer"),
                );
            }

            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }

            other => {
                eprintln!("Unknown option: {other}");
                print_help();
                std::process::exit(1);
            }
        }

        i += 1;
    }

    result
}

fn print_help() {
    println!(
        r#"generate-data options:

  --games N
      Number of games to generate
      default: {DEFAULT_NUM_GAMES}

  --output PATH
      Output dataset
      default: {DEFAULT_OUTPUT_PATH}

  --max-moves N
      Maximum moves per game
      default: {DEFAULT_MAX_MOVES}

  --cols N
      Board columns
      default: {DEFAULT_COLS}

  --rows N
      Board rows including hidden rows
      default: {DEFAULT_ROWS}

  --num-colors N
      Number of colors
      default: {DEFAULT_NUM_COLORS}

  --threads N
      Number of Rayon worker threads
      default: logical CPU count
"#
    );
}

struct GameResult {
    samples: Vec<Sample>,
    max_chain: u32,
    score: u32,
    moves: usize,
}

fn play_one_game(gc: GameConfig, max_moves: usize) -> GameResult {
    let evaluator = SimulationEvaluator;

    // 重要:
    // GameState::default() ではなく、
    // CLIから作ったGameConfigを利用する。
    let mut game = GameState::new(gc);

    let mut samples = Vec::<Sample>::with_capacity(max_moves);

    let mut move_count = 0usize;

    while game.phase != GamePhase::GameOver {
        if game.phase != GamePhase::Falling {
            break;
        }

        if move_count >= max_moves {
            break;
        }

        let current_piece = match &game.current_piece {
            Some(fp) => fp.piece,
            None => break,
        };

        let puyo_state = PuyoState {
            board: game.board.clone(),
            current: current_piece,
            next: game.next_piece,
            next_next: game.next_next_piece,
        };

        // 配置前の局面を教師データにする。
        let board_data = PuyoGame::encode_board(&puyo_state);

        let context_data = PuyoGame::encode_context(&puyo_state);

        // ヒューリスティック教師AI。
        let result = evaluator.find_best_move(&puyo_state);

        match result {
            Some((placement, score)) => {
                let action_index = placement_to_index(&placement) as u8;

                game.apply_placement(&placement);

                samples.push(Sample {
                    board_data,
                    context_data,
                    action_index,
                    value_target: score as f32,
                });

                move_count += 1;
            }

            None => break,
        }
    }

    GameResult {
        samples,
        max_chain: game.max_chain,
        score: game.score,
        moves: move_count,
    }
}

fn main() {
    let args = parse_args();

    let gc = GameConfig::new(args.cols, args.rows, args.num_colors);

    // PuyoGame trait側でも同じconfigを使わせる。
    // OnceLockなのでプロセス中に1度だけ設定する。
    puyo_player::puyo_game::init_config(gc);

    // output先ディレクトリを作る。
    if let Some(parent) = Path::new(&args.output_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("Failed to create output directory");
        }
    }

    // threadsを指定された場合だけ
    // Rayon global poolを設定する。
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to initialize Rayon thread pool");
    }

    let num_threads = rayon::current_num_threads();

    println!("Generating training data");

    println!(
        "Game config: cols={}, rows={}, num_colors={}",
        gc.cols, gc.rows, gc.num_colors
    );

    println!(
        "Games: {}, max_moves={}, threads={}",
        args.num_games, args.max_moves, num_threads
    );

    println!("Output: {}", args.output_path);

    let start_time = Instant::now();

    let games_done = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let total_chain_sum = AtomicU64::new(0);
    let total_max_chain = AtomicU64::new(0);
    let total_score = AtomicU64::new(0);

    // --------------------------------------------------
    // ゲーム単位で並列化
    // --------------------------------------------------

    let results: Vec<GameResult> = (0..args.num_games)
        .into_par_iter()
        .map(|_| {
            let result = play_one_game(gc, args.max_moves);

            let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;

            total_samples.fetch_add(result.samples.len() as u64, Ordering::Relaxed);

            total_chain_sum.fetch_add(result.max_chain as u64, Ordering::Relaxed);

            total_score.fetch_add(result.score as u64, Ordering::Relaxed);

            total_max_chain.fetch_max(result.max_chain as u64, Ordering::Relaxed);

            // ログがボトルネックにならないよう
            // 100ゲーム単位で表示。
            if done % 100 == 0 || done == args.num_games {
                let elapsed = start_time.elapsed().as_secs_f64();

                let games_per_sec = done as f64 / elapsed;

                let remaining = args.num_games - done;

                let eta = if games_per_sec > 0.0 {
                    remaining as f64 / games_per_sec
                } else {
                    0.0
                };

                let samples = total_samples.load(Ordering::Relaxed);

                let chain_sum = total_chain_sum.load(Ordering::Relaxed);

                let avg_chain = chain_sum as f64 / done as f64;

                let max_chain = total_max_chain.load(Ordering::Relaxed);

                let score_sum = total_score.load(Ordering::Relaxed);

                let avg_score = score_sum as f64 / done as f64;

                println!(
                    "[{:>6}/{}] \
                         samples: {:>8} | \
                         chain(max/avg): {}/{:.2} | \
                         avg_score: {:.0} | \
                         {:.2} games/s | \
                         ETA: {:.0}s",
                    done,
                    args.num_games,
                    samples,
                    max_chain,
                    avg_chain,
                    avg_score,
                    games_per_sec,
                    eta,
                );
            }

            result
        })
        .collect();

    // --------------------------------------------------
    // 並列処理後にDatasetをまとめる
    // --------------------------------------------------

    let mut dataset = Dataset::new();

    let mut final_max_chain = 0u32;
    let mut final_score_sum = 0u64;
    let mut final_move_sum = 0usize;

    for result in results {
        final_max_chain = final_max_chain.max(result.max_chain);

        final_score_sum += result.score as u64;

        final_move_sum += result.moves;

        dataset.samples.extend(result.samples);
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    let avg_score = final_score_sum as f64 / args.num_games as f64;

    let avg_moves = final_move_sum as f64 / args.num_games as f64;

    let games_per_sec = args.num_games as f64 / elapsed;

    println!();
    println!("Data generation complete");
    println!("Games       : {}", args.num_games);
    println!("Samples     : {}", dataset.samples.len());
    println!("Max chain   : {}", final_max_chain);
    println!("Avg score   : {:.1}", avg_score);
    println!("Avg moves   : {:.1}", avg_moves);
    println!("Elapsed     : {:.2}s", elapsed);
    println!("Throughput  : {:.2} games/s", games_per_sec);

    dataset
        .save(&args.output_path)
        .expect("Failed to save dataset");

    println!("Saved to {}", args.output_path);
}
