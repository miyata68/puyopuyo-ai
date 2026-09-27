//! Synchronous pre-tick decisions with independently seeded exploration.
use crate::pvp_data::*;
use az_framework::mcts::InferenceProvider;
use puyo_core::placement::placement_to_index;
use puyo_core::pvp::{MatchResult, MatchState, PlayerId};
use puyo_core::pvp_encoding::*;
use puyo_player::pvp_search::*;
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub search: PvpSearchConfig,
    pub max_ticks: u64,
    pub temperature: f32,
    pub temperature_drop_tick: u64,
    pub streams: [u8; 2],
}
impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            search: Default::default(),
            max_ticks: 2000,
            temperature: 1.,
            temperature_drop_tick: 100,
            streams: [0, 1],
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct MatchMetrics {
    pub result: MatchResult,
    pub truncated: bool,
    pub ticks: u64,
    pub samples: [usize; 2],
    pub max_chain: [u32; 2],
    pub garbage_sent: u64,
    pub garbage_offset: u64,
    pub garbage_dropped: u64,
    pub identical_action_ticks: u64,
    pub simultaneous_action_ticks: u64,
    pub symmetry_break_tick: Option<u64>,
    pub actions: Vec<[Option<usize>; 2]>,
}
impl MatchMetrics {
    pub fn identical_action_rate(&self) -> f64 {
        if self.simultaneous_action_ticks == 0 {
            0.
        } else {
            self.identical_action_ticks as f64 / self.simultaneous_action_ticks as f64
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct SelfPlayGame {
    pub samples: Vec<PvpAlphaZeroSample>,
    pub metrics: MatchMetrics,
    pub final_state: MatchState,
}
pub fn play_match(
    provider: &dyn InferenceProvider,
    seed: u64,
    config: &SelfPlayConfig,
) -> Result<SelfPlayGame, String> {
    config.search.validate()?;
    if config.streams[0] == config.streams[1]
        || config.streams.iter().any(|s| *s > 1)
        || !config.temperature.is_finite()
        || config.temperature < 0.
    {
        return Err("invalid self-play stream assignment/temperature".into());
    }
    let mut state = MatchState::new(seed);
    let mut records = Vec::new();
    let mut metrics = MatchMetrics {
        result: MatchResult::Ongoing,
        truncated: false,
        ticks: 0,
        samples: [0; 2],
        max_chain: [0; 2],
        garbage_sent: 0,
        garbage_offset: 0,
        garbage_dropped: 0,
        identical_action_ticks: 0,
        simultaneous_action_ticks: 0,
        symmetry_break_tick: None,
        actions: Vec::new(),
    };
    while state.result() == MatchResult::Ongoing && state.tick < config.max_ticks {
        // Immutable borrow of exactly the same root for both searches. No step
        // occurs until both results and both sampled actions have been collected.
        let root = &state;
        let mut actions = [None; 2];
        for id in [PlayerId::Player1, PlayerId::Player2] {
            if root.requires_action(id) {
                let stream = config.streams[id.index()];
                let result = PvpSearchV1::search(
                    root,
                    id,
                    provider,
                    &config.search,
                    derive_seed(seed, root.tick, stream, SeedPurpose::RootNoise),
                )?;
                let temperature = if root.tick < config.temperature_drop_tick {
                    config.temperature
                } else {
                    0.
                };
                actions[id.index()] = Some(result.select_action(
                    temperature,
                    derive_seed(seed, root.tick, stream, SeedPurpose::ActionSample),
                )?);
                records.push(MoveRecord {
                    board_data: pvp_board_to_tensor_data(root, id),
                    context_data: pvp_context_to_tensor_data(root, id),
                    improved_policy: result.policy.to_vec(),
                    perspective: id,
                    legal_mask: pvp_valid_action_mask(root, id).to_vec(),
                });
            }
        }
        if actions.iter().all(Option::is_some) {
            metrics.simultaneous_action_ticks += 1;
            if actions[0] == actions[1] {
                metrics.identical_action_ticks += 1;
            }
        }
        metrics
            .actions
            .push(actions.map(|a| a.map(|p| placement_to_index(&p))));
        let event = state
            .step(actions[0], actions[1])
            .map_err(|e| e.to_string())?;
        for e in event.players {
            metrics.garbage_sent += e.sent;
            metrics.garbage_offset += e.offset;
            metrics.garbage_dropped += e.dropped as u64;
        }
        if metrics.symmetry_break_tick.is_none()
            && (state.players[0].board != state.players[1].board
                || state.players[0].piece_index != state.players[1].piece_index)
        {
            metrics.symmetry_break_tick = Some(event.tick);
        }
    }
    metrics.result = state.result();
    metrics.truncated = state.result() == MatchResult::Ongoing;
    metrics.ticks = state.tick;
    metrics.max_chain = state.players.each_ref().map(|p| p.max_chain);
    let samples = finish_records(records, state.result());
    for s in &samples {
        s.validate()?;
        metrics.samples[s.perspective as usize] += 1;
    }
    Ok(SelfPlayGame {
        samples,
        metrics,
        final_state: state,
    })
}
