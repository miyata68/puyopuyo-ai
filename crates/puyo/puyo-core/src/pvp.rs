//! Deterministic simultaneous-tick versus engine, independent of the solo trainer.
use crate::board::{Board, PuyoColor};
use crate::config::GameConfig;
use crate::piece::{Piece, Placement};
use crate::placement::{enumerate_placements, place_piece_on_board};
use crate::rand::seeded_piece;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerId {
    Player1,
    Player2,
}
impl PlayerId {
    pub fn index(self) -> usize {
        match self {
            Self::Player1 => 0,
            Self::Player2 => 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerPhase {
    Ready,
    Chaining,
    Dead,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchResult {
    Ongoing,
    Player1Win,
    Player2Win,
    Draw,
}
impl MatchResult {
    pub fn winner(self) -> Option<PlayerId> {
        match self {
            Self::Player1Win => Some(PlayerId::Player1),
            Self::Player2Win => Some(PlayerId::Player2),
            _ => None,
        }
    }
    /// None until terminal; terminal value is always win/draw/loss, never score.
    pub fn outcome_for(self, player: PlayerId) -> Option<i8> {
        match self {
            Self::Ongoing => None,
            Self::Draw => Some(0),
            _ => Some(if self.winner() == Some(player) { 1 } else { -1 }),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TsuRules {
    pub garbage_rate: u32,
    pub max_garbage_per_drop: u32,
    pub all_clear_garbage: u32,
}
impl Default for TsuRules {
    fn default() -> Self {
        Self {
            garbage_rate: 70,
            max_garbage_per_drop: 30,
            all_clear_garbage: 30,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerState {
    pub board: Board,
    pub phase: PlayerPhase,
    pub piece_index: u64,
    pub pending_garbage: u64,
    pub confirmed_garbage: u64,
    pub attack_remainder: u32,
    pub all_clear_bonus: bool,
    pub score: u64,
    pub max_chain: u32,
    pub chain_count: u32,
    /// False after a drop: one placement must occur before another drop.
    pub garbage_drop_due: bool,
    pub garbage_drop_count: u64,
}
impl PlayerState {
    fn new() -> Self {
        Self {
            board: Board::new(&GameConfig::new(6, 14, 4)),
            phase: PlayerPhase::Ready,
            piece_index: 0,
            pending_garbage: 0,
            confirmed_garbage: 0,
            attack_remainder: 0,
            all_clear_bonus: false,
            score: 0,
            max_chain: 0,
            chain_count: 0,
            garbage_drop_due: true,
            garbage_drop_count: 0,
        }
    }
    fn dropping(&self) -> bool {
        self.phase == PlayerPhase::Ready && self.garbage_drop_due && self.confirmed_garbage > 0
    }
    fn attack_from_score(&mut self, score: u32, rules: TsuRules) -> u64 {
        let points = u64::from(score) + u64::from(self.attack_remainder);
        let mut attack = points / u64::from(rules.garbage_rate);
        self.attack_remainder = (points % u64::from(rules.garbage_rate)) as u32;
        // Consume on the next clear, even when its base score is below the rate.
        if self.all_clear_bonus {
            attack += u64::from(rules.all_clear_garbage);
            self.all_clear_bonus = false;
        }
        attack
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchState {
    pub players: [PlayerState; 2],
    pub tick: u64,
    pub seed: u64,
    pub result: MatchResult,
    rules: TsuRules,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchError {
    InvalidRules,
    InvalidBoard(PlayerId),
    Finished,
    MissingAction(PlayerId),
    UnexpectedAction(PlayerId),
    IllegalPlacement(PlayerId),
}
impl std::fmt::Display for MatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MatchError {}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlayerStepResult {
    pub placed: bool,
    pub chain_step: u32,
    pub score: u32,
    pub generated_attack: u64,
    /// Incoming canceled by this player's attack, including simultaneous attacks.
    pub offset: u64,
    pub sent: u64,
    pub dropped: u32,
    pub chain_finished: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchStepResult {
    pub tick: u64,
    pub players: [PlayerStepResult; 2],
    pub result: MatchResult,
}
impl MatchState {
    pub fn new(seed: u64) -> Self {
        Self::with_rules(seed, TsuRules::default()).expect("valid default rules")
    }
    pub fn with_rules(seed: u64, rules: TsuRules) -> Result<Self, MatchError> {
        if rules.garbage_rate == 0
            || rules.max_garbage_per_drop == 0
            || rules.max_garbage_per_drop > 30
        {
            return Err(MatchError::InvalidRules);
        }
        Ok(Self {
            players: [PlayerState::new(), PlayerState::new()],
            tick: 0,
            seed,
            result: MatchResult::Ongoing,
            rules,
        })
    }
    pub fn rules(&self) -> TsuRules {
        self.rules
    }
    pub fn result(&self) -> MatchResult {
        self.result
    }
    pub fn winner(&self) -> Option<PlayerId> {
        self.result.winner()
    }
    pub fn outcome_for(&self, player: PlayerId) -> Option<i8> {
        self.result.outcome_for(player)
    }
    pub fn piece_at(&self, index: u64) -> Piece {
        seeded_piece(self.seed, index, 4)
    }
    pub fn current_piece(&self, player: PlayerId) -> Piece {
        self.piece_at(self.players[player.index()].piece_index)
    }
    pub fn requires_action(&self, player: PlayerId) -> bool {
        let p = &self.players[player.index()];
        self.result == MatchResult::Ongoing && p.phase == PlayerPhase::Ready && !p.dropping()
    }
    /// Uses the existing canonical (same-color deduplicated) action space.
    pub fn legal_actions(&self, player: PlayerId) -> Vec<Placement> {
        if !self.requires_action(player) {
            return Vec::new();
        }
        enumerate_placements(
            &self.players[player.index()].board,
            &self.current_piece(player),
        )
    }
    /// All validation precedes mutation: errors leave the entire match unchanged.
    pub fn step(
        &mut self,
        p1: Option<Placement>,
        p2: Option<Placement>,
    ) -> Result<MatchStepResult, MatchError> {
        if self.result != MatchResult::Ongoing {
            return Err(MatchError::Finished);
        }
        let actions = [p1, p2];
        for id in [PlayerId::Player1, PlayerId::Player2] {
            if self.players[id.index()].board.config != GameConfig::new(6, 14, 4) {
                return Err(MatchError::InvalidBoard(id));
            }
            match (self.requires_action(id), actions[id.index()]) {
                (true, None) => return Err(MatchError::MissingAction(id)),
                (false, Some(_)) => return Err(MatchError::UnexpectedAction(id)),
                (true, Some(action)) if !self.legal_actions(id).contains(&action) => {
                    return Err(MatchError::IllegalPlacement(id))
                }
                _ => {}
            }
        }
        let mut events = [PlayerStepResult::default(); 2];
        // Each local update reads only its own start-of-tick state. No opponent state
        // is accessed until both local results exist; no board snapshot clones needed.
        for i in 0..2 {
            let piece = self.piece_at(self.players[i].piece_index);
            local_step(
                &mut self.players[i],
                actions[i],
                piece,
                self.rules,
                self.seed,
                self.tick,
                i,
                &mut events[i],
            );
        }
        resolve_attacks(&mut self.players, &mut events);
        // Pending belongs to the opponent's currently active chain.
        for i in 0..2 {
            if events[1 - i].chain_finished {
                self.players[i].confirmed_garbage = self.players[i]
                    .confirmed_garbage
                    .saturating_add(self.players[i].pending_garbage);
                self.players[i].pending_garbage = 0;
            }
        }
        self.result = match (
            self.players[0].phase == PlayerPhase::Dead,
            self.players[1].phase == PlayerPhase::Dead,
        ) {
            (true, true) => MatchResult::Draw,
            (true, false) => MatchResult::Player2Win,
            (false, true) => MatchResult::Player1Win,
            _ => MatchResult::Ongoing,
        };
        let tick = self.tick;
        self.tick = self.tick.wrapping_add(1);
        Ok(MatchStepResult {
            tick,
            players: events,
            result: self.result,
        })
    }
}
#[allow(clippy::too_many_arguments)]
fn local_step(
    p: &mut PlayerState,
    action: Option<Placement>,
    piece: Piece,
    rules: TsuRules,
    seed: u64,
    tick: u64,
    id: usize,
    event: &mut PlayerStepResult,
) {
    match p.phase {
        PlayerPhase::Dead => return,
        PlayerPhase::Ready if p.dropping() => {
            let count = p
                .confirmed_garbage
                .min(u64::from(rules.max_garbage_per_drop)) as u32;
            p.confirmed_garbage -= u64::from(count);
            let mut columns = [0, 1, 2, 3, 4, 5];
            let mut rng = mix(seed
                ^ mix(tick)
                ^ mix(id as u64 + 1)
                ^ mix(p.garbage_drop_count.wrapping_add(123)));
            for j in (1..6).rev() {
                rng = mix(rng.wrapping_add(0x9e3779b97f4a7c15));
                columns.swap(j, (rng % (j as u64 + 1)) as usize);
            }
            for n in 0..count {
                let col = columns[n as usize % 6];
                let row = p.board.column_height(col);
                if row >= p.board.config.rows || p.board.get(col, row).is_occupied() {
                    p.phase = PlayerPhase::Dead; // overflow, never overwrite a cell
                } else {
                    p.board.drop_puyo(col, PuyoColor::Garbage);
                    event.dropped += 1;
                }
            }
            p.garbage_drop_count = p.garbage_drop_count.wrapping_add(1);
            p.garbage_drop_due = false;
        }
        PlayerPhase::Ready => {
            if let Some(action) = action {
                place_piece_on_board(&mut p.board, &piece, &action);
                p.piece_index = p.piece_index.wrapping_add(1);
                p.garbage_drop_due = true;
                p.chain_count = 0;
                event.placed = true;
                if !p.board.find_clearable_groups().is_empty() {
                    p.phase = PlayerPhase::Chaining;
                }
            }
        }
        PlayerPhase::Chaining => {
            if let Some(score) = p.board.resolve_one_step(p.chain_count + 1) {
                p.chain_count += 1;
                p.max_chain = p.max_chain.max(p.chain_count);
                p.score = p.score.saturating_add(u64::from(score));
                event.chain_step = p.chain_count;
                event.score = score;
                event.generated_attack = p.attack_from_score(score, rules);
            }
            if p.board.find_clearable_groups().is_empty() {
                event.chain_finished = true;
                p.phase = PlayerPhase::Ready;
                p.garbage_drop_due = true;
                if p.chain_count > 0 && p.board.is_empty() {
                    p.all_clear_bonus = true;
                }
            }
        }
    }
    if p.phase == PlayerPhase::Ready && p.board.is_game_over() {
        p.phase = PlayerPhase::Dead;
    }
}
fn resolve_attacks(players: &mut [PlayerState; 2], events: &mut [PlayerStepResult; 2]) {
    let mut remaining = [events[0].generated_attack, events[1].generated_attack];
    for i in 0..2 {
        // Oldest confirmed incoming first, then the opponent's ongoing chain.
        for incoming in [
            &mut players[i].confirmed_garbage,
            &mut players[i].pending_garbage,
        ] {
            let canceled = remaining[i].min(*incoming);
            *incoming -= canceled;
            remaining[i] -= canceled;
            events[i].offset += canceled;
        }
    }
    let mutual = remaining[0].min(remaining[1]);
    for i in 0..2 {
        remaining[i] -= mutual;
        events[i].offset += mutual;
        events[i].sent = remaining[i];
    }
    for i in 0..2 {
        players[1 - i].pending_garbage =
            players[1 - i].pending_garbage.saturating_add(remaining[i]);
    }
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_conversion_and_remainder() {
        let mut p = PlayerState::new();
        let r = TsuRules::default();
        assert_eq!(p.attack_from_score(70, r), 1);
        assert_eq!(p.attack_remainder, 0);
        assert_eq!(p.attack_from_score(40, r), 0);
        assert_eq!(p.attack_remainder, 40);
        assert_eq!(p.attack_from_score(40, r), 1);
        assert_eq!(p.attack_remainder, 10);
    }
    #[test]
    fn symmetric_offset_with_existing_incoming() {
        let mut p = [PlayerState::new(), PlayerState::new()];
        p[0].confirmed_garbage = 3;
        p[0].pending_garbage = 8;
        p[1].confirmed_garbage = 2;
        let mut e = [
            PlayerStepResult {
                generated_attack: 20,
                ..Default::default()
            },
            PlayerStepResult {
                generated_attack: 7,
                ..Default::default()
            },
        ];
        let mut swapped = [p[1].clone(), p[0].clone()];
        let mut reversed = [e[1], e[0]];
        resolve_attacks(&mut p, &mut e);
        resolve_attacks(&mut swapped, &mut reversed);
        assert_eq!(p, [swapped[1].clone(), swapped[0].clone()]);
        assert_eq!(e, [reversed[1], reversed[0]]);
        assert_eq!(e[0].sent, 4);
        assert_eq!(p[1].pending_garbage, 4);
        assert_eq!(e[0].offset, 16);
    }
}
