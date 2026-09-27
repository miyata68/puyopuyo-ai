use crate::game::Game;

/// Trait for game evaluation strategies.
/// Generic over the game type to support different games.
pub trait Evaluator<G: Game> {
    /// AI最善手を探索する。
    fn find_best_move(&self, state: &G::State) -> Option<(G::Action, f64)>;

    /// MCTSシミュレーション数を変更する。対応していない評価器では何もしない。
    fn set_num_simulations(&mut self, _num_simulations: usize) {}
}
