use wasm_bindgen::prelude::*;

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

use puyo_core::config::GameConfig;
use puyo_core::game::{GamePhase, GameState};
use puyo_core::state::PuyoState;
use puyo_nn::model::{PuyoNet, PuyoNetConfig};
use puyo_player::eval::SimulationEvaluator;
use puyo_player::nn_eval::{MctsConfig, NnEvaluator};
use puyo_player::placement::enumerate_placements;
use puyo_player::puyo_game::{self, PuyoGame};
use puyo_player::Evaluator;

type InferBackend = NdArray;

#[wasm_bindgen]
pub struct WasmGame {
    state: GameState,
    config: GameConfig,
    evaluator: Box<dyn Evaluator<PuyoGame>>,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(cols: usize, rows: usize, num_colors: usize) -> WasmGame {
        let config = GameConfig::new(cols, rows, num_colors);
        puyo_game::init_config(config);
        WasmGame {
            state: GameState::new(config),
            config,
            evaluator: Box::new(SimulationEvaluator),
        }
    }

    /// ボードの列数。
    #[wasm_bindgen]
    pub fn board_cols(&self) -> u32 {
        self.config.cols as u32
    }

    /// ボードの行数（隠し行含む）。
    #[wasm_bindgen]
    pub fn board_rows(&self) -> u32 {
        self.config.rows as u32
    }

    /// 表示行数。
    #[wasm_bindgen]
    pub fn board_visible_rows(&self) -> u32 {
        self.config.visible_rows() as u32
    }

    /// 使用する色の数。
    #[wasm_bindgen]
    pub fn num_colors(&self) -> u32 {
        self.config.num_colors as u32
    }

    /// Load NN model weights from bytes. Returns true on success, false on failure.
    /// model_bytes: binary model data (BinBytesRecorder format)
    #[wasm_bindgen]
    pub fn load_nn_model(&mut self, model_bytes: &[u8]) -> bool {
        let device: <InferBackend as Backend>::Device = Default::default();
        let config = PuyoNetConfig::new().with_game_config(self.config);
        let record = match BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(model_bytes.to_vec(), &device)
        {
            Ok(r) => r,
            Err(_) => return false,
        };
        let model: PuyoNet<InferBackend> = config.init(&device).load_record(record);
        self.evaluator = Box::new(NnEvaluator::with_game_config(model, device, self.config));
        true
    }

    /// Load NN model with MCTS mode. Returns true on success, false on failure.
    /// model_bytes: binary model data (BinBytesRecorder format)
    /// num_simulations: number of MCTS simulations per move
    #[wasm_bindgen]
    pub fn load_nn_model_with_mcts(&mut self, model_bytes: &[u8], num_simulations: u32) -> bool {
        let device: <InferBackend as Backend>::Device = Default::default();
        let config = PuyoNetConfig::new().with_game_config(self.config);
        let record = match BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(model_bytes.to_vec(), &device)
        {
            Ok(r) => r,
            Err(_) => return false,
        };
        let model: PuyoNet<InferBackend> = config.init(&device).load_record(record);

        let mcts_config = MctsConfig {
            num_simulations: num_simulations as usize,
            ..MctsConfig::default()
        };

        self.evaluator = Box::new(
            NnEvaluator::with_game_config(model, device, self.config).with_mcts(mcts_config),
        );

        true
    }

    /// Update MCTS simulation count without reloading the model.
    #[wasm_bindgen]
    pub fn set_mcts_simulations(&mut self, num_simulations: u32) {
        self.evaluator.set_num_simulations(num_simulations as usize);
    }

    /// Switch back to heuristic evaluator.
    #[wasm_bindgen]
    pub fn use_heuristic(&mut self) {
        self.evaluator = Box::new(SimulationEvaluator);
    }

    /// Get the board as a flat Vec<u8>, column-major, bottom to top.
    /// Length = COLS * ROWS. Each byte is a PuyoColor (0=empty, 1..NUM_COLORS=colors).
    #[wasm_bindgen]
    pub fn get_board(&self) -> Vec<u8> {
        self.state.board.to_flat()
    }

    /// Get current piece info: [axis_color, sat_color, col, row_int, row_frac_x100, orientation]
    /// Returns empty vec if no current piece.
    #[wasm_bindgen]
    pub fn get_current_piece(&self) -> Vec<u8> {
        match self.state.get_current_piece_info() {
            Some((axis, sat, col, row, ori)) => {
                vec![
                    axis,
                    sat,
                    col,
                    row as u8,
                    ((row.fract()) * 100.0) as u8,
                    ori,
                ]
            }
            None => vec![],
        }
    }

    /// Get next piece info: [axis_color, sat_color]
    #[wasm_bindgen]
    pub fn get_next_piece(&self) -> Vec<u8> {
        let (axis, sat) = self.state.get_next_piece_info();
        vec![axis, sat]
    }

    /// Get next-next piece info: [axis_color, sat_color]
    #[wasm_bindgen]
    pub fn get_next_next_piece(&self) -> Vec<u8> {
        let (axis, sat) = self.state.get_next_next_piece_info();
        vec![axis, sat]
    }

    /// Get current score.
    #[wasm_bindgen]
    pub fn get_score(&self) -> u32 {
        self.state.score
    }

    /// Get max chain achieved.
    #[wasm_bindgen]
    pub fn get_max_chain(&self) -> u32 {
        self.state.max_chain
    }

    /// Get game phase: 0=Falling, 1=Resolving, 2=GameOver
    #[wasm_bindgen]
    pub fn get_phase(&self) -> u8 {
        self.state.phase.as_u8()
    }

    /// Get total pieces placed.
    #[wasm_bindgen]
    pub fn get_total_pieces(&self) -> u32 {
        self.state.total_pieces
    }

    /// Move left.
    #[wasm_bindgen]
    pub fn move_left(&mut self) -> bool {
        self.state.move_left()
    }

    /// Move right.
    #[wasm_bindgen]
    pub fn move_right(&mut self) -> bool {
        self.state.move_right()
    }

    /// Rotate clockwise.
    #[wasm_bindgen]
    pub fn rotate_cw(&mut self) -> bool {
        self.state.rotate_cw()
    }

    /// Rotate counter-clockwise.
    #[wasm_bindgen]
    pub fn rotate_ccw(&mut self) -> bool {
        self.state.rotate_ccw()
    }

    /// Hard drop. Returns the chain count (0 if no chain).
    #[wasm_bindgen]
    pub fn hard_drop(&mut self) -> u32 {
        match self.state.hard_drop() {
            Some(result) => result.chain_count,
            None => 0,
        }
    }

    /// Soft drop: move piece down by 1 cell. Returns true if moved, false if at landing position.
    #[wasm_bindgen]
    pub fn soft_drop(&mut self) -> bool {
        self.state.soft_drop()
    }

    /// Tick game with gravity. Returns chain count if piece landed and chains occurred.
    #[wasm_bindgen]
    pub fn tick(&mut self, gravity: f32) -> u32 {
        match self.state.tick(gravity) {
            Some(result) => result.chain_count,
            None => 0,
        }
    }

    /// AI: compute best move. Returns [col, orientation] or empty if no move.
    /// orientation: 0=North, 1=East, 2=South, 3=West
    #[wasm_bindgen]
    pub fn ai_best_move(&self) -> Vec<u8> {
        if self.state.phase != GamePhase::Falling {
            return vec![];
        }

        let current_piece = match &self.state.current_piece {
            Some(fp) => fp.piece,
            None => return vec![],
        };

        let puyo_state = PuyoState {
            board: self.state.board.clone(),
            current: current_piece,
            next: self.state.next_piece,
            next_next: self.state.next_next_piece,
        };

        let result = self.evaluator.find_best_move(&puyo_state);

        match result {
            Some((placement, score)) => {
                let mut bytes = vec![placement.col as u8, placement.orientation.as_u8()];
                bytes.extend_from_slice(&score.to_le_bytes());
                bytes
            }
            None => vec![],
        }
    }

    /// AI: compute and immediately apply the best move.
    /// Returns chain count from the placement.
    #[wasm_bindgen]
    pub fn ai_play_move(&mut self) -> u32 {
        if self.state.phase != GamePhase::Falling {
            return 0;
        }

        let current_piece = match &self.state.current_piece {
            Some(fp) => fp.piece,
            None => return 0,
        };

        let puyo_state = PuyoState {
            board: self.state.board.clone(),
            current: current_piece,
            next: self.state.next_piece,
            next_next: self.state.next_next_piece,
        };

        let result = self.evaluator.find_best_move(&puyo_state);

        match result {
            Some((placement, _score)) => {
                let chain_result = self.state.apply_placement(&placement);
                chain_result.chain_count
            }
            None => 0,
        }
    }

    /// Apply a placement directly by column and orientation.
    /// orientation: 0=North, 1=East, 2=South, 3=West
    /// Returns chain count from the placement.
    #[wasm_bindgen]
    pub fn apply_placement_direct(&mut self, col: u8, ori: u8) -> u32 {
        use puyo_core::piece::{Orientation, Placement};

        let orientation = match ori {
            0 => Orientation::North,
            1 => Orientation::East,
            2 => Orientation::South,
            3 => Orientation::West,
            _ => return 0,
        };
        let placement = Placement::new(col as usize, orientation);
        let result = self.state.apply_placement(&placement);
        result.chain_count
    }

    /// Enumerate all legal placements for the current piece.
    /// Returns a flat Vec<u8> of [col, orientation, col, orientation, ...].
    #[wasm_bindgen]
    pub fn enumerate_placements(&self) -> Vec<u8> {
        let current_piece = match &self.state.current_piece {
            Some(fp) => fp.piece,
            None => return vec![],
        };

        let placements = enumerate_placements(&self.state.board, &current_piece);
        let mut result = Vec::with_capacity(placements.len() * 2);
        for p in &placements {
            result.push(p.col as u8);
            result.push(p.orientation.as_u8());
        }
        result
    }

    /// Restart the game.
    #[wasm_bindgen]
    pub fn restart(&mut self) {
        self.state.restart();
    }

    /// Restart the game with a deterministic piece sequence.
    #[wasm_bindgen]
    pub fn restart_with_seed(&mut self, seed: u32) {
        self.state.restart_with_seed(seed as u64);
    }
}
