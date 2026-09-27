/// ターン制ゲームを抽象化するトレイト。
/// MCTS、NN学習、評価器がこのトレイトを通じてゲームロジックにアクセスする。
pub trait Game: Clone + Send + Sync + 'static {
    /// ゲームの状態（例: Board + ピースキュー）
    type State: Clone + Send + Sync;

    /// プレイヤーが取れるアクション（例: Placement）
    type Action: Clone + Copy + Send + Sync + std::fmt::Debug + Eq + std::hash::Hash;

    /// アクション適用結果（例: ChainResult）
    type ActionResult: Clone + Send + Sync;

    // --- 次元情報 ---

    /// アクション空間のサイズ（例: COLS * 4 = 12）
    fn num_actions() -> usize;

    /// 盤面テンソル形状: (channels, height, width)
    fn board_tensor_shape() -> (usize, usize, usize);

    /// コンテキストテンソルサイズ（例: ピースone-hot）
    fn context_tensor_size() -> usize;

    // --- 状態クエリ ---

    /// ゲームオーバーかどうか
    fn is_terminal(state: &Self::State) -> bool;

    // --- アクション列挙 ---

    /// 合法アクションの列挙
    fn legal_actions(state: &Self::State) -> Vec<Self::Action>;

    /// 有効アクションマスク（0..num_actions のインデックス）
    fn valid_action_mask(state: &Self::State) -> Vec<bool>;

    /// アクション → フラットインデックス
    fn action_to_index(action: &Self::Action) -> usize;

    /// フラットインデックス → アクション
    fn index_to_action(index: usize) -> Self::Action;

    // --- 状態遷移 ---

    /// アクションを適用し、(新状態, 結果) を返す
    fn apply_action(
        state: &Self::State,
        action: &Self::Action,
    ) -> (Self::State, Self::ActionResult);

    /// 結果から即時報酬を取得
    fn reward(result: &Self::ActionResult) -> f32;

    // --- NNエンコーディング ---

    /// 盤面をフラットf32テンソルにエンコード
    fn encode_board(state: &Self::State) -> Vec<f32>;

    /// コンテキスト（ピース等）をフラットf32テンソルにエンコード
    fn encode_context(state: &Self::State) -> Vec<f32>;

    // --- MCTS用ターン進行 ---

    /// MCTS展開時のターン進行（次ピース生成等）。
    /// apply_action後の状態に対して呼ぶ。
    fn advance_turn(state: &mut Self::State);
}
