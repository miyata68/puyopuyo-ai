//! Paired, noise-free model comparison using two providers at every search node.
use az_framework::mcts::InferenceProvider;
use puyo_core::{piece::Placement, placement::placement_to_index, pvp::*};
use puyo_player::pvp_search::*;

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub simulations: usize,
    pub max_ticks: u64,
}
impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            simulations: 64,
            max_ticks: 2000,
        }
    }
}
impl EvalConfig {
    pub fn search_config(&self) -> PvpSearchConfig {
        PvpSearchConfig {
            simulations: self.simulations,
            root_noise: false,
            ..Default::default()
        }
    }
}
/// Both real decisions are computed before the caller is allowed to step.
pub fn evaluation_actions(
    root: &MatchState,
    providers: [&dyn InferenceProvider; 2],
    config: &EvalConfig,
) -> Result<[Option<Placement>; 2], String> {
    let mut actions = [None; 2];
    for id in [PlayerId::Player1, PlayerId::Player2] {
        let i = id.index();
        if root.requires_action(id) {
            let result = PvpSearchV1::search_with_providers(
                root,
                id,
                providers[i],
                providers[1 - i],
                &config.search_config(),
                derive_seed(root.seed, root.tick, i as u8, SeedPurpose::SearchInternal),
            )?;
            actions[i] = Some(result.select_action(0., 0)?);
        }
    }
    Ok(actions)
}
#[derive(Debug, Clone, PartialEq)]
pub struct EvalGame {
    pub seed: u64,
    pub model_a_seat: PlayerId,
    pub result: MatchResult,
    pub ticks: u64,
    pub actions: Vec<[Option<usize>; 2]>,
    pub final_state: MatchState,
}
impl EvalGame {
    pub fn outcome_a(&self) -> Option<i8> {
        self.result.outcome_for(self.model_a_seat)
    }
}
pub fn evaluate_game(
    a: &dyn InferenceProvider,
    b: &dyn InferenceProvider,
    seed: u64,
    model_a_seat: PlayerId,
    config: &EvalConfig,
) -> Result<EvalGame, String> {
    config.search_config().validate()?;
    let providers = if model_a_seat == PlayerId::Player1 {
        [a, b]
    } else {
        [b, a]
    };
    let mut state = MatchState::new(seed);
    let mut trace = Vec::new();
    while state.result() == MatchResult::Ongoing && state.tick < config.max_ticks {
        let actions = evaluation_actions(&state, providers, config)?;
        trace.push(actions.map(|a| a.map(|p| placement_to_index(&p))));
        state
            .step(actions[0], actions[1])
            .map_err(|e| e.to_string())?;
    }
    Ok(EvalGame {
        seed,
        model_a_seat,
        result: state.result(),
        ticks: state.tick,
        actions: trace,
        final_state: state,
    })
}
#[derive(Debug, Clone, PartialEq)]
pub struct EvalPair {
    pub games: [EvalGame; 2],
}
pub fn evaluate_pair(
    a: &dyn InferenceProvider,
    b: &dyn InferenceProvider,
    seed: u64,
    config: &EvalConfig,
) -> Result<EvalPair, String> {
    Ok(EvalPair {
        games: [
            evaluate_game(a, b, seed, PlayerId::Player1, config)?,
            evaluate_game(a, b, seed, PlayerId::Player2, config)?,
        ],
    })
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScoreCounts {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub truncated: usize,
}
impl ScoreCounts {
    pub fn add(&mut self, outcome: Option<i8>) {
        match outcome {
            Some(1) => self.wins += 1,
            Some(-1) => self.losses += 1,
            Some(0) => self.draws += 1,
            None => self.truncated += 1,
            _ => unreachable!(),
        }
    }
    pub fn score(&self) -> f64 {
        self.wins as f64 + 0.5 * self.draws as f64
    }
    pub fn score_rate(&self) -> Option<f64> {
        let completed = self.wins + self.losses + self.draws;
        (completed > 0).then(|| self.score() / completed as f64)
    }
    pub fn reversed(self) -> Self {
        Self {
            wins: self.losses,
            losses: self.wins,
            ..self
        }
    }
}
impl std::fmt::Display for ScoreCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rate = self.score_rate().map_or("NA".into(), |r| format!("{r:.6}"));
        write!(
            f,
            "wins={} losses={} draws={} truncated={} score={:.1} score_rate={rate}",
            self.wins,
            self.losses,
            self.draws,
            self.truncated,
            self.score()
        )
    }
}
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EvalSummary {
    pub pairs: usize,
    pub a: ScoreCounts,
    pub a_as_seat: [ScoreCounts; 2],
    pub a_wins_both: usize,
    pub b_wins_both: usize,
    pub split: usize,
    pub containing_draw: usize,
    pub containing_truncation: usize,
}
impl EvalSummary {
    pub fn add_pair(&mut self, pair: &EvalPair) {
        self.pairs += 1;
        for game in &pair.games {
            self.a.add(game.outcome_a());
            self.a_as_seat[game.model_a_seat.index()].add(game.outcome_a());
        }
        let outcomes = pair.games.each_ref().map(|g| g.outcome_a());
        self.a_wins_both += usize::from(outcomes == [Some(1), Some(1)]);
        self.b_wins_both += usize::from(outcomes == [Some(-1), Some(-1)]);
        self.split +=
            usize::from(outcomes == [Some(1), Some(-1)] || outcomes == [Some(-1), Some(1)]);
        self.containing_draw += usize::from(outcomes.contains(&Some(0)));
        self.containing_truncation += usize::from(outcomes.contains(&None));
    }
    pub fn print(&self) {
        println!(
            "SUMMARY pairs={} games={} A_wins={} B_wins={} draws={} truncated={}",
            self.pairs,
            self.pairs * 2,
            self.a.wins,
            self.a.losses,
            self.a.draws,
            self.a.truncated
        );
        println!("Model A {}", self.a);
        println!("Model B {}", self.a.reversed());
        println!("A_as_p1 {}", self.a_as_seat[0]);
        println!("A_as_p2 {}", self.a_as_seat[1]);
        println!("PAIRED A_wins_both={} B_wins_both={} split_1_1={} pairs_containing_draw={} pairs_containing_truncation={}", self.a_wins_both, self.b_wins_both, self.split, self.containing_draw, self.containing_truncation);
    }
}
