//! Fixed-perspective PUCT against a cached pre-tick opponent policy baseline.
//! This is a best-response approximation, not a simultaneous Nash solver.
use az_framework::mcts::InferenceProvider;
use puyo_core::piece::Placement;
use puyo_core::placement::index_to_placement;
use puyo_core::pvp::{MatchState, PlayerId};
use puyo_core::pvp_encoding::*;

#[derive(Debug, Clone, Copy)]
pub enum SeedPurpose {
    RootNoise = 0,
    ActionSample = 1,
    SearchInternal = 2,
}
/// Stream assignment can be [0,1] or [1,0]; player identity is never NN input.
/// For fixed match/tick, the six (stream,purpose) domains are distinct before
/// the bijective finalizer, guaranteeing distinct P1/P2 seeds.
pub fn derive_seed(match_seed: u64, tick: u64, stream: u8, purpose: SeedPurpose) -> u64 {
    mix(mix(match_seed)
        .wrapping_add(mix(tick))
        .wrapping_add((stream as u64) * 3 + purpose as u64))
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
#[derive(Clone)]
pub struct SearchRng(u64);
impl SearchRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        mix(self.0)
    }
    pub fn uniform(&mut self) -> f64 {
        // Rounding the largest midpoint can produce 1.0; keep Gumbel logs finite.
        (((self.next_u64() >> 11) as f64 + 0.5) / 9007199254740992.0).min(1.0 - f64::EPSILON)
    }
}
#[derive(Debug, Clone)]
pub struct PvpSearchConfig {
    pub simulations: usize,
    pub c_puct: f32,
    pub root_noise: bool,
    pub noise_epsilon: f32,
    pub max_depth: usize,
    pub forced_tick_limit: usize,
}
impl Default for PvpSearchConfig {
    fn default() -> Self {
        Self {
            simulations: 64,
            c_puct: 1.5,
            root_noise: true,
            noise_epsilon: 0.25,
            max_depth: 128,
            forced_tick_limit: 128,
        }
    }
}
impl PvpSearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.simulations == 0
            || self.max_depth == 0
            || self.forced_tick_limit == 0
            || !self.c_puct.is_finite()
            || self.c_puct <= 0.
            || !self.noise_epsilon.is_finite()
            || !(0.0..=1.0).contains(&self.noise_epsilon)
        {
            return Err("invalid PvP search configuration".into());
        }
        Ok(())
    }
}
fn infer(
    provider: &dyn InferenceProvider,
    s: &MatchState,
    id: PlayerId,
) -> Result<(Vec<f32>, f32), String> {
    let (logits, value) = provider.infer(
        &pvp_board_to_tensor_data(s, id),
        &pvp_context_to_tensor_data(s, id),
    );
    if logits.len() != NUM_ACTIONS
        || logits.iter().any(|x| !x.is_finite())
        || !value.is_finite()
        || !(-1.00001..=1.00001).contains(&value)
    {
        return Err("invalid PvP inference: expected 24 finite logits and value in [-1,1]".into());
    }
    Ok((logits, value.clamp(-1., 1.)))
}
fn masked_softmax(logits: &[f32], mask: &[bool; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let max = (0..NUM_ACTIONS)
        .filter(|&i| mask[i])
        .map(|i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut p = [0.; NUM_ACTIONS];
    let mut sum = 0.;
    for i in 0..NUM_ACTIONS {
        if mask[i] {
            p[i] = (logits[i] - max).exp();
            sum += p[i];
        }
    }
    if sum > 0. {
        for x in &mut p {
            *x /= sum;
        }
    }
    p
}
/// Replaceable opponent strategy interface. It receives only the immutable
/// pre-tick state, never a self candidate action or its successor.
pub trait OpponentPolicy {
    fn action(
        &self,
        provider: &dyn InferenceProvider,
        state: &MatchState,
        opponent: PlayerId,
    ) -> Result<Option<Placement>, String>;
}
pub struct MaskedArgmax;
impl OpponentPolicy for MaskedArgmax {
    fn action(
        &self,
        provider: &dyn InferenceProvider,
        state: &MatchState,
        id: PlayerId,
    ) -> Result<Option<Placement>, String> {
        if !state.requires_action(id) {
            return Ok(None);
        }
        let mask = pvp_valid_action_mask(state, id);
        let (logits, _) = infer(provider, state, id)?;
        let mut best = None;
        for i in 0..NUM_ACTIONS {
            if mask[i] && best.is_none_or(|j| logits[i] > logits[j]) {
                best = Some(i);
            }
        }
        best.map(|i| Some(index_to_placement(i, COLS)))
            .ok_or_else(|| "actionable opponent has no legal move".into())
    }
}
pub fn advance_until_decision(
    s: &mut MatchState,
    id: PlayerId,
    provider: &dyn InferenceProvider,
    opponent_policy: &dyn OpponentPolicy,
    limit: usize,
) -> Result<usize, String> {
    let mut ticks = 0;
    while s.outcome_for(id).is_none() && !s.requires_action(id) {
        if ticks >= limit {
            return Err("forced tick safety limit reached".into());
        }
        let mut actions = [None; 2];
        let other = opponent(id);
        actions[other.index()] = opponent_policy.action(provider, s, other)?;
        s.step(actions[0], actions[1]).map_err(|e| e.to_string())?;
        ticks += 1;
    }
    Ok(ticks)
}
/// Immutable prepared decision: all self candidates use the same baseline.
#[derive(Clone)]
pub struct PreparedDecision {
    pub state: MatchState,
    pub perspective: PlayerId,
    pub opponent_action: Option<Placement>,
}
impl PreparedDecision {
    pub fn new(
        state: &MatchState,
        id: PlayerId,
        provider: &dyn InferenceProvider,
        policy: &dyn OpponentPolicy,
    ) -> Result<Self, String> {
        Ok(Self {
            state: state.clone(),
            perspective: id,
            opponent_action: policy.action(provider, state, opponent(id))?,
        })
    }
    pub fn step_candidate(&self, action: Placement) -> Result<MatchState, String> {
        let mut next = self.state.clone();
        let mut actions = [None; 2];
        actions[self.perspective.index()] = Some(action);
        actions[opponent(self.perspective).index()] = self.opponent_action;
        next.step(actions[0], actions[1])
            .map_err(|e| e.to_string())?;
        Ok(next)
    }
}
struct Node {
    decision: PreparedDecision,
    mask: [bool; NUM_ACTIONS],
    priors: [f32; NUM_ACTIONS],
    visits: u32,
    total_value: f32,
    children: [Option<usize>; NUM_ACTIONS],
    value: f32,
    terminal: bool,
}
impl Node {
    fn expand(
        s: MatchState,
        id: PlayerId,
        provider: &dyn InferenceProvider,
        policy: &dyn OpponentPolicy,
    ) -> Result<Self, String> {
        let terminal_value = s.outcome_for(id);
        let mask = pvp_valid_action_mask(&s, id);
        let (priors, value, baseline) = if let Some(z) = terminal_value {
            ([0.; NUM_ACTIONS], z as f32, None)
        } else {
            if !s.requires_action(id) || !mask.iter().any(|&x| x) {
                return Err("search node is not a decision".into());
            }
            let (logits, value) = infer(provider, &s, id)?;
            (
                masked_softmax(&logits, &mask),
                value,
                policy.action(provider, &s, opponent(id))?,
            )
        };
        Ok(Self {
            decision: PreparedDecision {
                state: s,
                perspective: id,
                opponent_action: baseline,
            },
            mask,
            priors,
            visits: 0,
            total_value: 0.,
            children: [None; NUM_ACTIONS],
            value,
            terminal: terminal_value.is_some(),
        })
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct PvpSearchResult {
    pub policy: [f32; NUM_ACTIONS],
    pub visits: [u32; NUM_ACTIONS],
    pub root_priors: [f32; NUM_ACTIONS],
    pub value: f32,
}
impl PvpSearchResult {
    pub fn select_action(&self, temperature: f32, seed: u64) -> Result<Placement, String> {
        if !temperature.is_finite() || temperature < 0. {
            return Err("invalid temperature".into());
        }
        if self.visits.iter().all(|&n| n == 0) {
            return Err("terminal search has no action".into());
        }
        let mut best = 0;
        for i in 1..NUM_ACTIONS {
            if self.visits[i] > self.visits[best] {
                best = i;
            }
        }
        if temperature == 0. {
            return Ok(index_to_placement(best, COLS));
        }
        let max = self.visits[best] as f64;
        let weights: Vec<_> = self
            .visits
            .iter()
            .map(|&n| {
                if n == 0 {
                    0.
                } else {
                    ((n as f64 / max).ln() / temperature as f64).exp()
                }
            })
            .collect();
        let mut draw = SearchRng::new(seed).uniform() * weights.iter().sum::<f64>();
        for (i, w) in weights.iter().enumerate() {
            draw -= w;
            if draw < 0. {
                return Ok(index_to_placement(i, COLS));
            }
        }
        Ok(index_to_placement(best, COLS))
    }
}
pub struct PvpSearchV1;
impl PvpSearchV1 {
    pub fn search(
        state: &MatchState,
        id: PlayerId,
        provider: &dyn InferenceProvider,
        config: &PvpSearchConfig,
        seed: u64,
    ) -> Result<PvpSearchResult, String> {
        Self::search_with_opponent(state, id, provider, config, seed, &MaskedArgmax)
    }
    pub fn search_with_opponent(
        state: &MatchState,
        id: PlayerId,
        provider: &dyn InferenceProvider,
        config: &PvpSearchConfig,
        seed: u64,
        opponent_policy: &dyn OpponentPolicy,
    ) -> Result<PvpSearchResult, String> {
        config.validate()?;
        let mut root = Node::expand(state.clone(), id, provider, opponent_policy)?;
        if root.terminal {
            return Ok(PvpSearchResult {
                policy: [0.; NUM_ACTIONS],
                visits: [0; NUM_ACTIONS],
                root_priors: root.priors,
                value: root.value,
            });
        }
        if config.root_noise {
            // A normalized Gumbel perturbation distribution mixed with the prior.
            // No new RNG/distribution dependency; deterministic stream per root.
            let mut rng = SearchRng::new(seed);
            let mut gumbels = [0.; NUM_ACTIONS];
            for (i, g) in gumbels.iter_mut().enumerate() {
                if root.mask[i] {
                    *g = (-(-rng.uniform().ln()).ln()) as f32;
                }
            }
            let noise = masked_softmax(&gumbels, &root.mask);
            for (i, p) in root.priors.iter_mut().enumerate() {
                *p = (1. - config.noise_epsilon) * *p + config.noise_epsilon * noise[i];
            }
        }
        let mut nodes = vec![root];
        for _ in 0..config.simulations {
            let mut node = 0;
            let mut path = vec![0];
            let value = loop {
                if nodes[node].terminal || path.len() > config.max_depth {
                    break nodes[node].value;
                }
                let parent = &nodes[node];
                let mut best = None;
                let mut best_score = f32::NEG_INFINITY;
                for a in 0..NUM_ACTIONS {
                    if parent.mask[a] {
                        let (n, q) = parent.children[a].map_or((0, 0.), |c| {
                            let n = nodes[c].visits;
                            (
                                n,
                                if n == 0 {
                                    0.
                                } else {
                                    nodes[c].total_value / n as f32
                                },
                            )
                        });
                        let score = q + config.c_puct
                            * parent.priors[a]
                            * ((parent.visits + 1) as f32).sqrt()
                            / (1 + n) as f32;
                        if score > best_score {
                            best = Some(a);
                            best_score = score;
                        }
                    }
                }
                let a = best.ok_or("no legal PUCT action")?;
                if let Some(child) = nodes[node].children[a] {
                    node = child;
                    path.push(node);
                } else {
                    let mut s = nodes[node]
                        .decision
                        .step_candidate(index_to_placement(a, COLS))?;
                    advance_until_decision(
                        &mut s,
                        id,
                        provider,
                        opponent_policy,
                        config.forced_tick_limit,
                    )?;
                    let child = Node::expand(s, id, provider, opponent_policy)?;
                    let v = child.value;
                    let index = nodes.len();
                    nodes.push(child);
                    nodes[node].children[a] = Some(index);
                    path.push(index);
                    break v;
                }
            };
            // Same perspective at every depth. No score reward, sign flip or discount.
            for n in path {
                nodes[n].visits += 1;
                nodes[n].total_value += value;
            }
        }
        let mut visits = [0; NUM_ACTIONS];
        let mut policy = [0.; NUM_ACTIONS];
        for a in 0..NUM_ACTIONS {
            if let Some(c) = nodes[0].children[a] {
                visits[a] = nodes[c].visits;
                policy[a] = visits[a] as f32 / config.simulations as f32;
            }
        }
        Ok(PvpSearchResult {
            policy,
            visits,
            root_priors: nodes[0].priors,
            value: nodes[0].total_value / nodes[0].visits as f32,
        })
    }
}
