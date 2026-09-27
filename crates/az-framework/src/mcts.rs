use crate::game::Game;

use crate::nn_eval::MctsConfig;

/// Trait for generating seeds used by Gumbel MCTS.
/// Implement this to control how randomness is injected into the search.
pub trait SeedProvider: Send + Sync {
    fn seed(&self) -> u64;
}

/// Default seed provider using current time (nanosecond precision).
/// On WASM targets (where `std::time::SystemTime` is unreliable),
/// falls back to an atomic counter mixed with a constant.
pub struct TimeSeedProvider;

impl TimeSeedProvider {
    fn splitmix64(s: u64) -> u64 {
        let s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        let s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
        s ^ (s >> 31)
    }
}

impl SeedProvider for TimeSeedProvider {
    fn seed(&self) -> u64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            TimeSeedProvider::splitmix64(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0x1234_5678_9abc_def0);
            TimeSeedProvider::splitmix64(COUNTER.fetch_add(1, Ordering::Relaxed))
        }
    }
}

/// Fixed seed provider for deterministic testing.
pub struct FixedSeedProvider(pub u64);

impl SeedProvider for FixedSeedProvider {
    fn seed(&self) -> u64 {
        self.0
    }
}

/// Trait abstracting NN inference so MCTS is backend-agnostic.
/// Implementations: `DirectInference` (single-sample, any burn backend),
/// `BatchedInference` (GPU batched via inference server).
pub trait InferenceProvider {
    /// Run the neural network on a single board+context input.
    /// Returns (logits[num_actions], value_transformed).
    /// The value is already inverse-transformed (raw cumulative reward scale).
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32);

    /// Batch inference. Returns Vec of (logits, value) for each input.
    /// Default implementation calls `infer()` N times sequentially.
    fn infer_batch(&self, batch: &[(Vec<f32>, Vec<f32>)]) -> Vec<(Vec<f32>, f32)> {
        batch.iter().map(|(b, c)| self.infer(b, c)).collect()
    }
}

/// A node in the MCTS tree (Decision Node).
struct MctsNode<G: Game> {
    /// Number of visits.
    visit_count: u32,
    /// Sum of values from all visits (for computing Q = total_value / visit_count).
    total_value: f32,
    /// Prior probability from the policy network.
    prior: f32,
    /// NN policy priors for each action (set when expanded).
    priors: Vec<f32>,
    /// Raw NN logits before softmax (set when expanded).
    logits: Vec<f32>,
    /// Children indexed by action (0..num_actions-1). None = not yet expanded for this action.
    children: Vec<Option<usize>>,
    /// Whether this node has been expanded (network evaluated).
    expanded: bool,
    /// Terminal node (game over or no turns left).
    terminal: bool,
    /// Immediate reward received when transitioning TO this node (chain score).
    immediate_reward: f32,
    /// Game state at this node.
    state: G::State,
    /// Cached valid action mask (derived from state, immutable per node).
    valid_mask: Vec<bool>,
    /// Depth from root (root = 0).
    depth: u32,
    /// Virtual loss count for batched MCTS (number of in-flight simulations passing through this node).
    virtual_loss: u32,
}

/// MCTS tree.
pub struct MctsTree<G: Game> {
    nodes: Vec<MctsNode<G>>,
    root: usize,
    /// Discount factor for future rewards.
    gamma: f32,
    /// Minimum Q value observed in the tree (for Min-Max normalization).
    min_value: f32,
    /// Maximum Q value observed in the tree (for Min-Max normalization).
    max_value: f32,
    /// Value head output for the root node (used for completed Q-values).
    root_value: f32,
}

impl<G: Game> MctsTree<G> {
    /// Create a new MCTS tree rooted at the given game state.
    /// `gamma` is the discount factor for future rewards.
    pub fn new(state: &G::State, gamma: f32) -> Self {
        let num_actions = G::num_actions();
        let valid_mask = G::valid_action_mask(state);
        let root = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            priors: vec![0.0; num_actions],
            logits: vec![0.0; num_actions],
            children: vec![None; num_actions],
            expanded: false,
            terminal: G::is_terminal(state),
            immediate_reward: 0.0,
            state: state.clone(),
            valid_mask,
            depth: 0,
            virtual_loss: 0,
        };
        MctsTree {
            nodes: vec![root],
            root: 0,
            gamma,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            root_value: 0.0,
        }
    }

    /// Expand the root node and return the value estimate.
    fn expand_root(&mut self, provider: &dyn InferenceProvider) -> f32 {
        let v = self.expand_node(self.root, provider);
        self.root_value = v;
        self.nodes[self.root].visit_count += 1;
        v
    }

    /// Run one simulation forcing a specific action at the root, then PUCT for the rest.
    fn simulate_from_root_action(
        &mut self,
        action: usize,
        provider: &dyn InferenceProvider,
        c_puct_init: f32,
        c_puct_base: f32,
    ) {
        let child_id = self.get_or_create_child(self.root, action);
        let mut path: Vec<(usize, usize)> = vec![(self.root, action)];
        let mut node_id = child_id;

        // From child onward, use standard PUCT selection
        while self.nodes[node_id].expanded && !self.nodes[node_id].terminal {
            let act = self.select_action(node_id, c_puct_init, c_puct_base);
            path.push((node_id, act));
            node_id = self.get_or_create_child(node_id, act);
        }

        // Expand leaf
        let value = if self.nodes[node_id].terminal {
            0.0
        } else if !self.nodes[node_id].expanded {
            self.expand_node(node_id, provider)
        } else {
            0.0
        };

        // Backpropagate: compute cumulative values (immutable)
        let backups: Vec<f32> = path
            .iter()
            .rev()
            .scan(value, |acc, &(parent_id, act)| {
                if let Some(cid) = self.nodes[parent_id].children[act] {
                    *acc = self.nodes[cid].immediate_reward + self.gamma * *acc;
                }
                Some(*acc)
            })
            .collect();

        // Backpropagate: apply updates (mutable)
        for (&(parent_id, act), backup) in path.iter().rev().zip(&backups) {
            if let Some(cid) = self.nodes[parent_id].children[act] {
                self.min_value = self.min_value.min(*backup);
                self.max_value = self.max_value.max(*backup);
                self.nodes[cid].visit_count += 1;
                self.nodes[cid].total_value += *backup;
            }
        }
        self.nodes[self.root].visit_count += 1;
    }

    /// Normalize a Q value to [0, 1] using min-max normalization (MuZero Reanalyze).
    /// Returns 0.5 when min == max (no information yet).
    fn normalize_q(&self, q: f32) -> f32 {
        let range = self.max_value - self.min_value;
        if range > f32::EPSILON {
            ((q - self.min_value) / range).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Compute dynamic c_puct based on parent visit count.
    /// c(s) = log((1 + N(s) + c_base) / c_base) + c_init
    fn dynamic_c_puct(parent_visits: f32, c_puct_init: f32, c_puct_base: f32) -> f32 {
        ((1.0 + parent_visits + c_puct_base) / c_puct_base).ln() + c_puct_init
    }

    /// Select action using PUCT score with Min-Max normalized Q values and dynamic c_puct.
    fn select_action(&self, node_id: usize, c_puct_init: f32, c_puct_base: f32) -> usize {
        let node = &self.nodes[node_id];
        let parent_visits = node.visit_count.max(1) as f32;
        let sqrt_parent = parent_visits.sqrt();
        let c_puct = Self::dynamic_c_puct(parent_visits, c_puct_init, c_puct_base);

        node.valid_mask
            .iter()
            .enumerate()
            .filter(|(_, &is_valid)| is_valid)
            .map(|(action, _)| {
                let prior = node.priors[action];
                let (q, n) = match node.children[action] {
                    Some(child_id) => {
                        let child = &self.nodes[child_id];
                        // Include virtual loss: inflates visit count without adding value,
                        // making Q pessimistic to discourage concurrent selection.
                        let effective_n = child.visit_count + child.virtual_loss;
                        let q = if effective_n > 0 {
                            child.total_value / effective_n as f32
                        } else {
                            0.0
                        };
                        (q, effective_n as f32)
                    }
                    None => (0.0, 0.0),
                };
                let q_normalized = self.normalize_q(q);
                let puct = q_normalized + c_puct * prior * sqrt_parent / (1.0 + n);
                (action, puct)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(action, _)| action)
            .unwrap_or(0)
    }

    /// Get existing child or create a new one by simulating the action.
    fn get_or_create_child(&mut self, parent_id: usize, action: usize) -> usize {
        if let Some(child_id) = self.nodes[parent_id].children[action] {
            return child_id;
        }
        let child_id = self.create_child(parent_id, action);
        self.nodes[parent_id].children[action] = Some(child_id);
        child_id
    }

    /// Create a child node by simulating an action.
    fn create_child(&mut self, parent_id: usize, action: usize) -> usize {
        let num_actions = G::num_actions();
        let parent_state = &self.nodes[parent_id].state;
        let game_action = G::index_to_action(action);

        // Simulate placement
        let (mut new_state, action_result) = G::apply_action(parent_state, &game_action);

        let terminal = G::is_terminal(&new_state);
        let immediate_reward = G::reward(&action_result);

        // Advance turn (generate next piece etc.)
        G::advance_turn(&mut new_state);

        let parent_depth = self.nodes[parent_id].depth;
        let valid_mask = G::valid_action_mask(&new_state);
        let child = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: self.nodes[parent_id].priors[action],
            priors: vec![0.0; num_actions],
            logits: vec![0.0; num_actions],
            children: vec![None; num_actions],
            expanded: false,
            terminal,
            immediate_reward,
            state: new_state,
            valid_mask,
            depth: parent_depth + 1,
            virtual_loss: 0,
        };

        let child_id = self.nodes.len();
        self.nodes.push(child);
        child_id
    }

    /// Expand a node: run the neural network and set priors + logits for valid actions.
    fn expand_node(&mut self, node_id: usize, provider: &dyn InferenceProvider) -> f32 {
        let state = &self.nodes[node_id].state;

        let board_data = G::encode_board(state);
        let context_data = G::encode_context(state);

        let (logits_vec, v) = provider.infer(&board_data, &context_data);

        // Compute masked softmax for priors
        let priors = masked_softmax(&logits_vec, &self.nodes[node_id].valid_mask);

        // Store logits and priors
        let num_actions = G::num_actions();
        let mut stored_logits = vec![0.0f32; num_actions];
        for (i, logit) in logits_vec.iter().enumerate().take(num_actions) {
            stored_logits[i] = *logit;
        }
        self.nodes[node_id].logits = stored_logits;

        // Update priors of already-created children before moving priors into node
        for (action, &prior) in priors.iter().enumerate() {
            if let Some(child_id) = self.nodes[node_id].children[action] {
                self.nodes[child_id].prior = prior;
            }
        }
        self.nodes[node_id].priors = priors;

        self.nodes[node_id].expanded = true;

        v
    }

    /// Get Q values (average cumulative reward) for root's direct children.
    pub fn root_q_values(&self) -> Vec<f32> {
        let num_actions = G::num_actions();
        let root = &self.nodes[self.root];
        (0..num_actions)
            .map(|action| {
                root.children[action]
                    .map(|child_id| &self.nodes[child_id])
                    .filter(|child| child.visit_count > 0)
                    .map(|child| child.total_value / child.visit_count as f32)
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Run Sequential Halving over the considered actions, then spend remaining budget.
    /// Returns the final completed Q-values to avoid redundant recomputation.
    fn sequential_halving(
        &mut self,
        considered: &mut Vec<usize>,
        scores: &mut [f32],
        gumbels: &[f32],
        remaining_budget: usize,
        provider: &dyn InferenceProvider,
        config: &MctsConfig,
    ) -> Vec<f32> {
        let root_logits = self.nodes[self.root].logits.clone();
        let mask = self.nodes[self.root].valid_mask.clone();

        if remaining_budget == 0 {
            return compute_completed_q(self, &mask);
        }

        let num_phases = {
            let mut phases = 0u32;
            let mut n = considered.len();
            while n > 1 {
                n = n.div_ceil(2);
                phases += 1;
            }
            phases.max(1) as usize
        };

        let mut budget_used = 0usize;

        for phase in 0..num_phases {
            if considered.len() <= 1 {
                break;
            }

            let n_actions = considered.len();
            let budget_remaining = remaining_budget.saturating_sub(budget_used);
            let phases_left = num_phases - phase;
            let sims_per_action = (budget_remaining / (phases_left * n_actions)).max(1);

            budget_used += self.run_simulations(
                considered,
                budget_used,
                remaining_budget,
                sims_per_action,
                provider,
                config.c_puct_init,
                config.c_puct_base,
            );

            let q_completed = compute_completed_q(self, &mask);
            let sigma_bar =
                compute_sigma_bar(self, &q_completed, &mask, considered, config.c_visit);

            for &a in considered.iter() {
                scores[a] = gumbels[a] + root_logits[a] + sigma_bar[a];
            }

            let keep = n_actions.div_ceil(2);
            considered.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            considered.truncate(keep);
        }

        // Spend remaining budget on surviving action(s)
        self.run_simulations(
            considered,
            budget_used,
            remaining_budget,
            usize::MAX,
            provider,
            config.c_puct_init,
            config.c_puct_base,
        );

        compute_completed_q(self, &mask)
    }

    /// Run simulations on the considered actions, up to `sims_per_action` each,
    /// respecting the total budget. Returns the number of simulations run.
    #[allow(clippy::too_many_arguments)]
    fn run_simulations(
        &mut self,
        considered: &[usize],
        budget_used: usize,
        total_budget: usize,
        sims_per_action: usize,
        provider: &dyn InferenceProvider,
        c_puct_init: f32,
        c_puct_base: f32,
    ) -> usize {
        let mut count = 0usize;
        for &a in considered {
            for _ in 0..sims_per_action {
                if budget_used + count >= total_budget {
                    return count;
                }
                self.simulate_from_root_action(a, provider, c_puct_init, c_puct_base);
                count += 1;
            }
        }
        count
    }

    // --- Batched MCTS methods (virtual loss) ---

    /// Select a path from a root action down to an unexpanded/terminal leaf.
    /// Creates child nodes along the way. Does NOT expand or backpropagate.
    /// Returns (path, leaf_node_id).
    fn select_path_to_leaf(
        &mut self,
        root_action: usize,
        c_puct_init: f32,
        c_puct_base: f32,
    ) -> (Vec<(usize, usize)>, usize) {
        let child_id = self.get_or_create_child(self.root, root_action);
        let mut path: Vec<(usize, usize)> = vec![(self.root, root_action)];
        let mut node_id = child_id;

        while self.nodes[node_id].expanded && !self.nodes[node_id].terminal {
            let act = self.select_action(node_id, c_puct_init, c_puct_base);
            path.push((node_id, act));
            node_id = self.get_or_create_child(node_id, act);
        }

        (path, node_id)
    }

    /// Add virtual loss along a path to discourage concurrent selection.
    fn apply_virtual_loss(&mut self, path: &[(usize, usize)], weight: u32) {
        for &(parent_id, action) in path {
            if let Some(child_id) = self.nodes[parent_id].children[action] {
                self.nodes[child_id].virtual_loss += weight;
            }
        }
    }

    /// Remove virtual loss after backpropagation.
    fn remove_virtual_loss(&mut self, path: &[(usize, usize)], weight: u32) {
        for &(parent_id, action) in path {
            if let Some(child_id) = self.nodes[parent_id].children[action] {
                self.nodes[child_id].virtual_loss =
                    self.nodes[child_id].virtual_loss.saturating_sub(weight);
            }
        }
    }

    /// Backpropagate a value along a path, updating visit counts and values.
    fn backpropagate(&mut self, path: &[(usize, usize)], leaf_value: f32) {
        let backups: Vec<f32> = path
            .iter()
            .rev()
            .scan(leaf_value, |acc, &(parent_id, act)| {
                if let Some(cid) = self.nodes[parent_id].children[act] {
                    *acc = self.nodes[cid].immediate_reward + self.gamma * *acc;
                }
                Some(*acc)
            })
            .collect();

        for (&(parent_id, act), backup) in path.iter().rev().zip(&backups) {
            if let Some(cid) = self.nodes[parent_id].children[act] {
                self.min_value = self.min_value.min(*backup);
                self.max_value = self.max_value.max(*backup);
                self.nodes[cid].visit_count += 1;
                self.nodes[cid].total_value += *backup;
            }
        }
        self.nodes[self.root].visit_count += 1;
    }

    /// Encode the game state at a leaf node for NN inference.
    fn encode_leaf_state(&self, node_id: usize) -> (Vec<f32>, Vec<f32>) {
        let state = &self.nodes[node_id].state;
        (G::encode_board(state), G::encode_context(state))
    }

    /// Apply pre-computed NN results to a node. Returns the value.
    /// If the node was already expanded by a prior batch element, returns 0.0.
    fn expand_node_with_result(&mut self, node_id: usize, logits_vec: Vec<f32>, value: f32) -> f32 {
        if self.nodes[node_id].expanded {
            return 0.0;
        }

        let priors = masked_softmax(&logits_vec, &self.nodes[node_id].valid_mask);

        let num_actions = G::num_actions();
        let mut stored_logits = vec![0.0f32; num_actions];
        for (i, logit) in logits_vec.iter().enumerate().take(num_actions) {
            stored_logits[i] = *logit;
        }
        self.nodes[node_id].logits = stored_logits;

        for (action, &prior) in priors.iter().enumerate() {
            if let Some(child_id) = self.nodes[node_id].children[action] {
                self.nodes[child_id].prior = prior;
            }
        }
        self.nodes[node_id].priors = priors;
        self.nodes[node_id].expanded = true;

        value
    }

    /// Run a batch of simulations with virtual loss.
    /// Each element of `actions` specifies a root action to force.
    /// Leaves are collected and evaluated in a single batch NN call.
    fn simulate_batch(
        &mut self,
        actions: &[usize],
        provider: &dyn InferenceProvider,
        c_puct_init: f32,
        c_puct_base: f32,
    ) {
        // Phase 1: Select paths and apply virtual loss
        struct PendingSim {
            path: Vec<(usize, usize)>,
            leaf_id: usize,
            needs_inference: bool,
        }

        let mut pending: Vec<PendingSim> = Vec::with_capacity(actions.len());
        let mut inference_inputs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        // Maps from inference_inputs index to pending index
        let mut inference_map: Vec<usize> = Vec::new();

        for &action in actions {
            let (path, leaf_id) = self.select_path_to_leaf(action, c_puct_init, c_puct_base);
            self.apply_virtual_loss(&path, 1);

            let needs_inference = !self.nodes[leaf_id].terminal && !self.nodes[leaf_id].expanded;
            if needs_inference {
                let (board, ctx) = self.encode_leaf_state(leaf_id);
                inference_map.push(pending.len());
                inference_inputs.push((board, ctx));
            }
            pending.push(PendingSim {
                path,
                leaf_id,
                needs_inference,
            });
        }

        // Phase 2: Batch NN inference (all leaves at once)
        let inference_results = if !inference_inputs.is_empty() {
            provider.infer_batch(&inference_inputs)
        } else {
            vec![]
        };

        // Phase 3: Expand nodes and backpropagate, then remove virtual loss
        let mut infer_idx = 0;
        for sim in &pending {
            let value = if sim.needs_inference {
                let map_idx = inference_map[infer_idx];
                debug_assert_eq!(map_idx, infer_idx); // inference_map is sequential here
                let (logits, v) = inference_results[infer_idx].clone();
                infer_idx += 1;
                self.expand_node_with_result(sim.leaf_id, logits, v)
            } else {
                0.0
            };

            self.backpropagate(&sim.path, value);
            self.remove_virtual_loss(&sim.path, 1);
        }
    }

    /// Run simulations in batches, distributing budget across considered actions.
    /// Returns the number of simulations actually run.
    #[allow(clippy::too_many_arguments)]
    fn run_simulations_batched(
        &mut self,
        considered: &[usize],
        budget_used: usize,
        total_budget: usize,
        sims_per_action: usize,
        provider: &dyn InferenceProvider,
        c_puct_init: f32,
        c_puct_base: f32,
        num_leaves: usize,
    ) -> usize {
        // Build the full list of (action, repetition) pairs, same order as run_simulations
        let mut action_queue: Vec<usize> = Vec::new();
        for &a in considered {
            for _ in 0..sims_per_action {
                if budget_used + action_queue.len() >= total_budget {
                    break;
                }
                action_queue.push(a);
            }
            if budget_used + action_queue.len() >= total_budget {
                break;
            }
        }

        // Process in batches
        let mut count = 0;
        for chunk in action_queue.chunks(num_leaves) {
            self.simulate_batch(chunk, provider, c_puct_init, c_puct_base);
            count += chunk.len();
        }
        count
    }

    /// Sequential Halving with batched simulations.
    fn sequential_halving_batched(
        &mut self,
        considered: &mut Vec<usize>,
        scores: &mut [f32],
        gumbels: &[f32],
        remaining_budget: usize,
        provider: &dyn InferenceProvider,
        config: &MctsConfig,
    ) -> Vec<f32> {
        let root_logits = self.nodes[self.root].logits.clone();
        let mask = self.nodes[self.root].valid_mask.clone();

        if remaining_budget == 0 {
            return compute_completed_q(self, &mask);
        }

        let num_phases = {
            let mut phases = 0u32;
            let mut n = considered.len();
            while n > 1 {
                n = n.div_ceil(2);
                phases += 1;
            }
            phases.max(1) as usize
        };

        let mut budget_used = 0usize;

        for phase in 0..num_phases {
            if considered.len() <= 1 {
                break;
            }

            let n_actions = considered.len();
            let budget_remaining = remaining_budget.saturating_sub(budget_used);
            let phases_left = num_phases - phase;
            let sims_per_action = (budget_remaining / (phases_left * n_actions)).max(1);

            budget_used += self.run_simulations_batched(
                considered,
                budget_used,
                remaining_budget,
                sims_per_action,
                provider,
                config.c_puct_init,
                config.c_puct_base,
                config.num_leaves,
            );

            let q_completed = compute_completed_q(self, &mask);
            let sigma_bar =
                compute_sigma_bar(self, &q_completed, &mask, considered, config.c_visit);

            for &a in considered.iter() {
                scores[a] = gumbels[a] + root_logits[a] + sigma_bar[a];
            }

            let keep = n_actions.div_ceil(2);
            considered.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            considered.truncate(keep);
        }

        // Spend remaining budget on surviving action(s)
        self.run_simulations_batched(
            considered,
            budget_used,
            remaining_budget,
            usize::MAX,
            provider,
            config.c_puct_init,
            config.c_puct_base,
            config.num_leaves,
        );

        compute_completed_q(self, &mask)
    }
}

/// Compute masked softmax over logits.
fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let num_actions = mask.len();
    // Find max for numerical stability
    let max_logit = (0..num_actions)
        .filter(|&i| mask[i])
        .filter_map(|i| logits.get(i).copied())
        .fold(f32::NEG_INFINITY, f32::max);

    if max_logit == f32::NEG_INFINITY {
        return vec![0.0f32; num_actions]; // No valid actions
    }

    // Detect NaN/Inf in valid logits (indicates NN output corruption)
    for (i, &m) in mask.iter().enumerate().take(num_actions) {
        if m {
            let v = logits.get(i).copied().unwrap_or(0.0);
            assert!(
                v.is_finite(),
                "masked_softmax: logit[{i}] is {v} (NaN or Inf detected)"
            );
        }
    }

    let mut result: Vec<f32> = (0..num_actions)
        .map(|i| {
            if mask[i] {
                (logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = result.iter().sum();
    if sum > 0.0 {
        result.iter_mut().for_each(|r| *r /= sum);
    }

    result
}

/// Simple xorshift64-based PRNG returning f64 in [0, 1).
fn xorshift64_f64(state: &mut u64, mix: u64) -> f64 {
    let mut s = (*state).wrapping_add(mix.wrapping_mul(2654435761));
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // Guard against zero state (xorshift gets stuck at 0)
    if s == 0 {
        s = 0x5a17a453cc79b7d1;
    }
    *state = s;
    (s >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Sample from the standard Gumbel(0,1) distribution: g = -log(-log(u)).
fn sample_gumbel(state: &mut u64, mix: u64) -> f32 {
    let u = xorshift64_f64(state, mix).clamp(1e-20, 1.0 - 1e-10);
    -((-(u.ln())).ln()) as f32
}

/// Compute completed Q-values for all valid root actions.
/// Visited actions use actual Q from tree; unvisited actions use root value estimate.
fn compute_completed_q<G: Game>(tree: &MctsTree<G>, mask: &[bool]) -> Vec<f32> {
    let num_actions = G::num_actions();
    let root = &tree.nodes[tree.root];
    (0..num_actions)
        .map(|a| {
            if !mask[a] {
                return 0.0;
            }
            match root.children[a] {
                Some(child_id) if tree.nodes[child_id].visit_count > 0 => {
                    tree.nodes[child_id].total_value / tree.nodes[child_id].visit_count as f32
                }
                _ => tree.root_value,
            }
        })
        .collect()
}

/// Min-max normalize Q-values to [0,1].
/// Only indices where `mask[a]` is true are considered for min/max range.
/// Returns 0.0 for masked-out actions, 0.5 when all considered values are equal.
fn normalize_q_minmax(q_values: &[f32], mask: &[bool]) -> Vec<f32> {
    let num_actions = mask.len();
    let mut min_q = f32::INFINITY;
    let mut max_q = f32::NEG_INFINITY;
    for a in 0..num_actions {
        if mask[a] {
            min_q = min_q.min(q_values[a]);
            max_q = max_q.max(q_values[a]);
        }
    }
    let q_range = max_q - min_q;

    (0..num_actions)
        .map(|a| {
            if !mask[a] {
                return 0.0;
            }
            if q_range > f32::EPSILON {
                (q_values[a] - min_q) / q_range
            } else {
                0.5
            }
        })
        .collect()
}

/// Compute the improved policy target from logits and completed Q-values.
/// π_improved(a) ∝ π(a) · exp(advantage(a) · c_visit)
///
/// Q-values are min-max normalized to [0,1] before computing advantages,
/// so that c_visit operates on a consistent scale regardless of
/// the raw reward magnitude (following Gumbel MuZero paper assumptions).
fn compute_improved_policy(
    logits: &[f32],
    q_completed: &[f32],
    mask: &[bool],
    c_visit: f32,
) -> Vec<f32> {
    let num_actions = mask.len();
    let q_normalized = normalize_q_minmax(q_completed, mask);

    // Compute V_mixed: prior-weighted sum of normalized Q-values
    let priors = masked_softmax(logits, mask);
    let v_mixed: f32 = (0..num_actions)
        .filter(|&a| mask[a])
        .map(|a| priors[a] * q_normalized[a])
        .sum();

    // Compute improved logits: logit(a) + advantage(a) * c_visit
    let mut improved_logits = vec![f32::NEG_INFINITY; num_actions];
    for a in 0..num_actions {
        if mask[a] {
            let advantage = q_normalized[a] - v_mixed;
            improved_logits[a] = logits[a] + advantage * c_visit;
        }
    }

    masked_softmax(&improved_logits, mask)
}

/// Compute sigma_bar for Sequential Halving score updates.
/// sigma_bar(a) = (c_visit + N_max) * q_normalized(a)
/// where q_normalized is min-max normalized completed Q-value.
fn compute_sigma_bar<G: Game>(
    tree: &MctsTree<G>,
    q_completed: &[f32],
    mask: &[bool],
    considered: &[usize],
    c_visit: f32,
) -> Vec<f32> {
    let num_actions = G::num_actions();
    // Find max visit count among root children
    let root = &tree.nodes[tree.root];
    let n_max: f32 = considered
        .iter()
        .filter_map(|&a| root.children[a].map(|cid| tree.nodes[cid].visit_count as f32))
        .fold(0.0f32, f32::max);

    // Build mask restricted to considered actions for normalization range
    let mut considered_mask = vec![false; num_actions];
    for &a in considered {
        considered_mask[a] = mask[a];
    }
    let q_normalized = normalize_q_minmax(q_completed, &considered_mask);

    (0..num_actions)
        .map(|a| {
            if !mask[a] {
                return 0.0;
            }
            (c_visit + n_max) * q_normalized[a]
        })
        .collect()
}

/// Run Gumbel MCTS search using Sequential Halving with Gumbel noise.
///
/// Returns (improved_policy, q_values).
/// Seed is generated internally via `config.seed_provider`.
pub fn mcts_search<G: Game>(
    state: &G::State,
    provider: &dyn InferenceProvider,
    config: &MctsConfig,
) -> (Vec<f32>, Vec<f32>) {
    let seed = config.seed_provider.seed();
    let num_actions = G::num_actions();
    let mut tree = MctsTree::<G>::new(state, config.gamma);

    // 1. Expand root (1 NN evaluation)
    if config.num_simulations == 0 || tree.nodes[tree.root].terminal {
        let mask = &tree.nodes[tree.root].valid_mask;
        return (
            masked_softmax(&vec![0.0f32; num_actions], mask),
            vec![0.0; num_actions],
        );
    }
    tree.expand_root(provider);

    let mask = tree.nodes[tree.root].valid_mask.clone();
    let root_logits = tree.nodes[tree.root].logits.clone();

    // Collect valid actions
    let valid_actions: Vec<usize> = (0..num_actions).filter(|&a| mask[a]).collect();
    if valid_actions.is_empty() {
        return (vec![0.0; num_actions], vec![0.0; num_actions]);
    }
    if valid_actions.len() == 1 {
        let mut policy = vec![0.0f32; num_actions];
        policy[valid_actions[0]] = 1.0;
        return (policy, tree.root_q_values());
    }

    // 2. Sample Gumbel noise and compute initial scores: g(a) + logit(a)
    let mut rng_state = seed.wrapping_add(0xdeadbeef);
    let mut gumbels = vec![0.0f32; num_actions];
    let mut scores = vec![f32::NEG_INFINITY; num_actions];
    for &a in &valid_actions {
        let g = sample_gumbel(&mut rng_state, a as u64);
        gumbels[a] = g;
        scores[a] = g + root_logits[a];
    }

    // 3. Select top-m actions by initial score
    let m = config.m.min(valid_actions.len());
    let mut considered = valid_actions;
    considered.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    considered.truncate(m);

    // 4. Sequential Halving + spend remaining budget
    let remaining_budget = config.num_simulations.saturating_sub(1); // root expansion used 1
    let q_completed = tree.sequential_halving(
        &mut considered,
        &mut scores,
        &gumbels,
        remaining_budget,
        provider,
        config,
    );

    // 5. Compute improved policy target
    let improved_policy =
        compute_improved_policy(&root_logits, &q_completed, &mask, config.c_visit);

    (improved_policy, tree.root_q_values())
}

/// Run Gumbel MCTS with batched inference and virtual loss.
///
/// Same algorithm as `mcts_search`, but selects multiple leaves per iteration
/// and evaluates them in a single batch NN call. Virtual loss ensures
/// different leaves are selected within each batch.
///
/// `config.num_leaves` controls how many leaves are evaluated per NN call.
/// When `num_leaves == 1`, behavior is equivalent to `mcts_search`.
///
/// Returns (improved_policy, q_values).
/// Seed is generated internally via `config.seed_provider`.
pub fn mcts_search_batched<G: Game>(
    state: &G::State,
    provider: &dyn InferenceProvider,
    config: &MctsConfig,
) -> (Vec<f32>, Vec<f32>) {
    let seed = config.seed_provider.seed();
    let num_actions = G::num_actions();
    let mut tree = MctsTree::<G>::new(state, config.gamma);

    // 1. Expand root (1 NN evaluation)
    if config.num_simulations == 0 || tree.nodes[tree.root].terminal {
        let mask = &tree.nodes[tree.root].valid_mask;
        return (
            masked_softmax(&vec![0.0f32; num_actions], mask),
            vec![0.0; num_actions],
        );
    }
    tree.expand_root(provider);

    let mask = tree.nodes[tree.root].valid_mask.clone();
    let root_logits = tree.nodes[tree.root].logits.clone();

    // Collect valid actions
    let valid_actions: Vec<usize> = (0..num_actions).filter(|&a| mask[a]).collect();
    if valid_actions.is_empty() {
        return (vec![0.0; num_actions], vec![0.0; num_actions]);
    }
    if valid_actions.len() == 1 {
        let mut policy = vec![0.0f32; num_actions];
        policy[valid_actions[0]] = 1.0;
        return (policy, tree.root_q_values());
    }

    // 2. Sample Gumbel noise and compute initial scores: g(a) + logit(a)
    let mut rng_state = seed.wrapping_add(0xdeadbeef);
    let mut gumbels = vec![0.0f32; num_actions];
    let mut scores = vec![f32::NEG_INFINITY; num_actions];
    for &a in &valid_actions {
        let g = sample_gumbel(&mut rng_state, a as u64);
        gumbels[a] = g;
        scores[a] = g + root_logits[a];
    }

    // 3. Select top-m actions by initial score
    let m = config.m.min(valid_actions.len());
    let mut considered = valid_actions;
    considered.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    considered.truncate(m);

    // 4. Sequential Halving with batched simulations
    let remaining_budget = config.num_simulations.saturating_sub(1);
    let q_completed = tree.sequential_halving_batched(
        &mut considered,
        &mut scores,
        &gumbels,
        remaining_budget,
        provider,
        config,
    );

    // 5. Compute improved policy target
    let improved_policy =
        compute_improved_policy(&root_logits, &q_completed, &mask, config.c_visit);

    (improved_policy, tree.root_q_values())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_softmax_basic() {
        let num_actions = 12;
        let mut logits = vec![0.0f32; num_actions];
        logits[0] = 1.0;
        logits[1] = 2.0;
        logits[2] = 3.0;
        let mut mask = vec![false; num_actions];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let result = masked_softmax(&logits, &mask);

        // Highest logit (index 2) should have highest probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
        // Masked actions should be 0
        assert_eq!(result[3], 0.0);

        // Sum of valid actions should be ~1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_gumbel_finite() {
        let mut state = 12345u64;
        for i in 0..100 {
            let g = sample_gumbel(&mut state, i);
            assert!(g.is_finite(), "Gumbel sample should be finite");
        }
    }

    #[test]
    fn test_compute_improved_policy_normalized() {
        let num_actions = 12;
        let mut logits = vec![0.0f32; num_actions];
        logits[0] = 1.0;
        logits[1] = 2.0;
        logits[2] = 3.0;
        let mut q = vec![0.0f32; num_actions];
        q[0] = 10.0;
        q[1] = 20.0;
        q[2] = 15.0;
        let mut mask = vec![false; num_actions];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let policy = compute_improved_policy(&logits, &q, &mask, 5.0);

        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Improved policy should sum to 1.0, got {}",
            sum
        );
        // Action 1 has highest Q, should have highest improved probability
        assert!(policy[1] > policy[0]);
        assert!(policy[1] > policy[2]);
    }
}
