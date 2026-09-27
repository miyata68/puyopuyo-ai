# PvP AlphaZero V1

## 目的と互換性

Tick型対戦エンジンの上に、対戦専用NN、探索、自己対局データ、勝敗教師による学習を追加する。soloのPuyoState/PuyoGame/PuyoNet、GameConfigの次元定義、soloモデルの記録構造は維持する。PvPモデル、dataset、binaryは別物で、soloのcheckpointを読み込まない。

エンジンのルール、共通ツモ列、二回転、再配分、相殺、Tick時間は14-pvp-tick-engine.mdを参照。GUI、WASM対戦、Elo、実時間、Nash solverは本実装の対象外。

## ObservationとPerspective

`puyo_core::pvp_encoding`で固定サイズの入力を生成する。常にSelf→Opponent順。P2から見るとP2がSelfになる。左右の盤面反転は行わない。

board形状は`[10,14,6]`、flat indexは`channel*84 + row*6 + col`（row0が底）。

| channel | 内容 |
|---|---|
| 0,1,2,3 | Self Red, Green, Blue, Yellow |
| 4 | Self Garbage |
| 5,6,7,8 | Opponent Red, Green, Blue, Yellow |
| 9 | Opponent Garbage |

contextは**68要素**。下表の終端indexは包含。

| index | 内容・正規化 |
|---|---|
| 0–23 | Self current/next/next_next。各組はaxis4色one-hot→satellite4色one-hot |
| 24–47 | Opponentの同じ3組。Opponent自身のpiece_indexを基準 |
| 48,49 | Self pending, confirmed。`min(x,300)/30` |
| 50,51 | Opponent pending, confirmed。同上 |
| 52,53 | Self/Opponent attack_remainder / garbage_rate |
| 54,55 | Self/Opponent all_clear_bonus、0/1 |
| 56–59 | Self phase: ActionableReady, Chaining, GarbageDropping, Dead |
| 60–63 | Opponent phase、同順 |
| 64,65 | Self/Opponent `min(chain_count,20)/20` |
| 66,67 | Self/Opponent garbage_drop_due、0/1 |

各組の内部順はRed,Green,Blue,Yellow。Readyかつdrop_dueかつconfirmed>0はGarbageDroppingとして表す。それ以外のReadyはActionableReady。終端状態では探索がNNを呼ばない。

入力に絶対PlayerId、tick、累積score、max_chainは入れない。MatchState自体が保持するseed/indexは遷移の決定論性に使用するが、NNには上記の3組だけを渡す。

## NN・推論

`puyo_nn::pvp_model::PvpPuyoNet` / `PvpPuyoNetConfig` を追加。構造:

- stem 3×3 convolution、10ch→64ch、ReLU。
- FiLM generator: context68→128→`2*64*6`。
- 既存ResidualBlockを再利用した6ブロック。Conv/BatchNorm/ReLU/Conv/BatchNorm/FiLM/skip/ReLU。
- policy: 1×1 convolution→2ch、BatchNorm/ReLU、flatten→24 logits。
- value: 1×1 convolution→1ch、BatchNorm/ReLU、flatten→64→ReLU→1→**tanh**。

出力Valueは入力Perspectiveの勝ち+1、引分0、負け-1。soloのvalue_transform/inverse_transform/VALUE_SCALEは使用しない。`PvpGameModel<B>` のpostprocess_valueはidentityで、既存DirectInferenceとinference_serverに接続できる。

モデルは基底パス`X`に対し`X.bin`と`X.config.json`。metadataはmodel_kind=pvp、format_version=1、board_channels=10、rows=14、cols=6、num_colors=4、context_size=68、num_actions=24と6つのarchitectureパラメータを記録。kindや次元・versionの不一致、metadata欠落、片方だけ存在するモデルは明示的エラー。黙ってランダムへfallbackしない。完全に未存在の場合のみmodel_init_seedで1個の共有モデルを初期化し、self-playが初期モデルを保存する。

## 同時着手と探索

既存のGame trait/MCTSは1 actionで状態を更新し得点報酬を返す一人用なので、PvPには`PvpSearchV1`を使用する。

V1は**opponent-policy best-response approximation**。厳密なSM-MCTS/Nash探索ではない。木全体でPerspectiveを固定し、PUCTでSelf行動だけを選ぶ。相手行動は`OpponentPolicy` interfaceの`MaskedArgmax`で予測する。

ノードの状態はSelfの次のdecision状態。合法maskは`MatchState::legal_actions()`から既存action indexへ変換し、Tsu二回転と同色組重複除去を反映。invalidはprior/visit targetとも0。Action数は24。

各ノード展開時、**候補を適用する前の同じTick開始状態**から相手Perspectiveのpolicyを求め、masked argmaxをキャッシュする。`PreparedDecision::step_candidate`は全Self候補に同じ相手Actionを組み合わせてstepする。相手の推論へ候補適用後の状態は渡さない。OpponentPolicyにはSelf候補引数を持たせない。

遷移後は`advance_until_decision()`でSelfが次にAction可能になるまでforced Tickを進める。Self連鎖中/落下中はNone、相手がAction可能なら毎Tickの開始状態から相手policyを求める。両者Chainingならstep(None,None)。terminalで停止、128 forced Tick超はエラー。

ノードはstate/perspective、mask、priors、visits、total_value、children、leaf value、terminalを保持する。ノードは生成時に展開済みなのでexpanded boolは不要。標準64 simulations、c_puct=1.5、最大decision深さ128。深さ上限ではNN評価を返す。terminalはoutcome_for(perspective)。バックアップは同じValueをそのまま戻し、符号反転・割引・即時score報酬はない。root教師policyは`N(a)/sum(N)`。

## 完全対称ロックと探索乱数

同じ盤面・共通ツモ・NN・Self/Opp正規化・決定論的選択ではP1/P2が永久に同じ手を選ぶ可能性がある。ゲームルール、NN入力へのID、異なるモデル、異なるツモでは対称性を崩さない。

自己対局には独立した探索stream A/B（0/1）を割り当てる。通常はP1=A/P2=B、`--swap-streams true`でP1=B/P2=A。seed導出は:

```text
mix(mix(match_seed) + mix(tick) + stream*3 + purpose)
```

加算はu64 wrapping、mixはSplitMix64 finalizer。purposeはRootNoise=0、ActionSample=1、SearchInternal=2。固定match/tickでstream/purposeの組は異なる値になり、bijective mixerによってP1/P2のseedを区別できる。乱数生成器も状態をseedからのみ初期化するSplitMix64。時刻やthread_rngは使わない。

rootでは合法手ごとにGumbel乱数を生成し、softmaxでnoise分布を作る。`0.75*NN prior + 0.25*noise`で混合する。Dirichletではなく、追加の分布/RNG依存を不要にするGumbel-based explorationを選択した。Dirichlet alpha=0.30とは分布が異なり、その同一性を主張しない。無効化は`--root-noise false`。

Action選択は序盤`N(a)^(1/temperature)`を正規化した分布から別purposeの乱数でsampling。温度1.0、100Tick以降0が既定。温度0は最大visitを選ぶ。評価は`--root-noise false --temperature 0`。同じ手になる偶然は許す。異なる手を強制しない。

自己対局はTick開始stateを不変参照して両方のsearchを完了させ、その後一度だけstepする。P1の実行ActionをP2 searchへ見せない。

## Datasetと打ち切り

`PvpAlphaZeroDataset`は8byte magic `PUYOPVP1`とbincode v1のmetadata/samples。metadataにformat_version/model_kind/board_channels/rows/cols/context_size/num_actionsを持つ。soloファイルを拒否し、load/save時に次元・finite値・policy合計・invalid確率0・target -1/0/+1を検証。

各Action可能playerからboard_data、context_data、improved_policy、legal_mask、perspectiveを記録。perspectiveの絶対IDはラベル確定・診断用の記録だけに残り、NN入力には渡さない。試合終了後outcome_for(record.perspective)をvalue_targetとする。score/連鎖点/生存時間/おじゃま報酬は足さない。

max_ticks既定2000。到達時はエンジンのMatchResultを変更せず、self-playでTruncated扱いにする。その試合の全sampleを破棄する。timeoutをDraw=0教師にすると決着回避を評価するおそれがあるため。Truncation rate>1%でWARNING。

4色の24通りの同一置換を両盤面の通常色4chと双方の3ツモ48要素へ適用する。Garbageの4/9ch、context48以降、policy、legal mask、Valueは保持する。

## 学習

`train-pvp`は別binary。policy CE＋value_loss_weight×raw Value MSE。重み既定1.0は[-1,1]の勝敗誤差を標準の係数1で学習する初期設定で、CLI変更可能。softmaxから除外するのはlegal_mask=falseだけであり、未訪問の合法手（教師確率0）は除外しない。

Adam、learning_rate=0.001、1000steps、batch128を既定とする。seed付きsamplingでバッチと色置換を選択。学習後はvalid()モデルとmetadataを保存する。新モデルから次のself-playを実行可能。optimizer状態は保存せず、再開時はAdam状態を初期化する。

## CLI（PowerShell）

CPUは既定。以下は通常規模の構成（64ch×6blocks）:

```powershell
Set-Location C:\work\puyopuyo\puyopuyo-ai
cargo run --release -p puyo-trainer --no-default-features --bin pvp-self-play -- `
  --games 100 --simulations 64 --model-path artifacts/pvp/puyo_pvp_model `
  --output data/pvp/pvp_alphazero_iter_001.bin --seed-offset 100000 `
  --max-ticks 2000 --temperature 1.0 --temperature-drop-tick 100 `
  --root-noise true --model-init-seed 42 --backend cpu

cargo run --release -p puyo-trainer --no-default-features --bin train-pvp -- `
  --data data/pvp/pvp_alphazero_iter_001.bin `
  --model-path artifacts/pvp/puyo_pvp_model `
  --output-model artifacts/pvp/puyo_pvp_model_next `
  --steps 1000 --batch-size 128 --value-loss-weight 1.0 --backend cpu
```

CUDA利用時は`--no-default-features`を外し`--backend cuda`へ変更。self-playは`--threads`と`--inference-batch-size`を指定でき、既存のinference_serverで共有モデルの推論を集約する。既定threads=1。CPUの小規模smokeには未作成モデルパスで`--residual-channels 8 --residual-blocks 1 --games 32 --simulations 8 --verify-replay true`を指定する。既存モデルのarchitectureはmetadataから読む。

## 診断と再現性

ログはcompleted/P1 wins/P2 wins/draws/truncated/rate、平均Tick、両者sample数、平均と最大の試合内max_chain、おじゃま送信/相殺/落下、同時Action数・一致数・一致率、symmetry_break_tickの平均を出す。symmetry_break_tickは初めて盤面またはpiece_indexが不一致となったstepの0-based Tick番号。95%以上の試合が50Tick超の間対称ならWARNING。

`--verify-replay true`は各試合を同じmodel/seed/configで再実行し、Action列・全最終state・samples・metricsを厳密比較する。CPU/1threadで全32試合の再現を実測。異なるハードウェアやGPU backend、動的バッチサイズの浮動小数点演算までbitwise同一という保証はない。同じmodel、config、backend、実行条件を再現条件とする。

## 将来SM-MCTSへ変更する箇所

`OpponentPolicy`とノードのcached baselineをmixed strategy、同時手ノード、regret matching/Nash等へ置換する。現在は決定論的な相手policyへのbest response近似のため、戦略的な混合や均衡を学習できる保証はない。
streamのA/B割当は交換可能なのでpaired/mirrored evaluationに利用できる。ただし既存エンジンのおじゃま端数配置はplayer_idもseedに含むため、探索stream交換だけで全試合の勝敗が厳密に反転するとは限らない。今回Elo等は実装しない。

## 検証

encoding、固定Value、masked policy、64simulation、same-tick baseline、forced Tick、seed分離、対称fixture、stream交換、再現、教師値、打ち切り除外、色置換、metadata拒否、NN形状/値域、CE/MSEをテストする。CUDAなし探索テスト:

```powershell
cargo test -p puyo-player --features nn --test pvp_search
```

workspace全体のfmt/check/testと既存pvp_smokeも回帰対象。
