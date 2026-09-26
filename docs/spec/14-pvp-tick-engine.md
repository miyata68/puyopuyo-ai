# PvP Tick Engine

## 目的と範囲

`puyo_core::pvp` は決定論的な2人対戦エンジン。既存の `GameState`、`PuyoState`、一人用trainer、NN、MCTS、WASM/UIは従来の用途のまま残す。
盤面は専用の `GameConfig::new(6, 14, 4)`。表示12行、隠し2行、通常色4色、2個組、同色4個消去。既存の小盤面用defaultは変更しない。

**1 chain step = opponent can place one piece** は研究用の離散時間近似であり、実機フレーム完全再現ではない。相手自身もChaining、落下Tick、Deadなら配置できない。

## 型とAPI

- `PlayerId::{Player1, Player2}`: 配列indexの型付き指定。
- `PlayerPhase::{Ready, Chaining, Dead}`。
- `MatchResult::{Ongoing, Player1Win, Player2Win, Draw}`。
- `MatchState`: `players: [PlayerState; 2]`, `tick: u64`, `seed: u64`, `result: MatchResult`, 非公開の `rules: TsuRules`。
- `PlayerState`: `board`, `phase`, `piece_index: u64`, `pending_garbage: u64`, `confirmed_garbage: u64`, `attack_remainder: u32`, `all_clear_bonus: bool`, `score: u64`, `max_chain: u32`, `chain_count: u32`, `garbage_drop_due: bool`, `garbage_drop_count: u64`。
- `TsuRules`: `garbage_rate: u32 = 70`, `max_garbage_per_drop: u32 = 30`, `all_clear_garbage: u32 = 30`。`with_rules` はレート0、落下上限0または30超をエラーにする。
- `MatchState::new(seed)` / `with_rules(seed, rules)`。
- `requires_action(id)` / `legal_actions(id)` / `current_piece(id)` / `piece_at(index)`。
- `step(Option<Placement>, Option<Placement>) -> Result<MatchStepResult, MatchError>`。
- `result()` / `winner()` / `outcome_for(id)`。進行中のoutcomeはNone、終了後は勝ち+1、引分0、負け-1。scoreはValueに使わない。

`MatchStepResult` は処理したTick番号、両者のイベント、最終勝敗を返す。イベントには配置、連鎖段数、得点、攻撃生成数、相殺数、送信数、実落下数、連鎖終了フラグがある。同時攻撃の相殺数は双方にそれぞれ記録する。

公開フィールドは盤面fixture/探索用途にも使えるが、呼び出し側で整合性を保持する必要がある。進行中にルールを変更するAPIはない。外部から盤面configを変更した場合stepはInvalidBoardを返す。通常はnewとstepで遷移させる。

## Tickの処理順

1. Tick開始状態で双方の入力を検証。必要なAction欠落、不要なAction指定、不正配置、終了済みのstepを明示的エラーにする。エラーでは状態を一切変更しない。
2. 各プレイヤーのローカル更新を計算。自分の盤面と状態だけを読み、相手の変更済み状態は参照しない。
3. 双方の攻撃量が揃ってから、自分のincoming、同時攻撃の順に相殺し、余剰を相手のpendingへ追加。
4. このTickで終了した連鎖から相手に向かう残存pendingをconfirmedへ移す。
5. 両者の死亡状態から勝敗決定。同時死亡はDraw。Tickを1増やす。

更新は入力検証後に行い、独立したローカル状態とイベントで2-phase処理する。Tickごとの盤面snapshot cloneは不要。MatchState自体はClone/PartialEq/Eqで探索・再現比較可能。

## 配置・連鎖の時間

Readyは1組を配置する。消去可能グループができればChainingへ移るが、配置Tickには消去しない。
Chainingは公開した `Board::resolve_one_step(chain_count + 1)` を1回だけ呼ぶ。消去と重力の後に次段の有無を調べ、その場で最終段ならReadyに戻す。終了確認だけの余分なTickはない。
既存 `resolve_chains()` は同じ処理をループして最後まで解決する。一人用の挙動は維持する。配置は既存 `place_piece_on_board` を共有。

5連鎖対3連鎖では最初の3Tickは双方が1段ずつ消去、4・5Tick目は後者が配置できる（その配置で自身が再発火した場合は当然Chainingになる）。
PvPの `MatchState::legal_actions()` は `enumerate_tsu_placements()` を使用する。通の二回転（double rotation）により、左右が高い列や壁で塞がれていてもSouthを合法にする。対象列への到達可能性、縦2個の配置高さ、軸が最上段隠し行へ着地しない制約は維持する。フレーム単位の2回の入力操作は再現しない。一人用の `enumerate_placements()` は二回転なしの従来仕様を維持し、内部共通関数のフラグで切り替える。同色組の重複除去を使用し、合法Actionはその列挙の正規形。indexは `col * 4 + orientation`（North=0, East=1, South=2, West=3）のまま。

## ツモ列

既存 `seeded_piece(match_seed, piece_index, 4)` を共有する。各自のpiece_indexは配置した時だけ1増える。時間、thread_rng、グローバル乱数状態に依存しない。連鎖・おじゃま落下中にはindexが進まない。

## おじゃまの盤面表現

`PuyoColor::Garbage = 5` を追加。既存0〜4の値は維持。
`is_color()`/`is_normal_color()`は通常色のみ、`is_garbage()`はおじゃま、`is_occupied()`は空以外。
高さ・重力・衝突・配置判定は占有を使用。連結グループ探索と得点は通常色のみ。
通常色の消去セルに上下左右で直接隣接するおじゃまだけを消す。おじゃま経由で消去は伝播しない。隠し行にある直接隣接おじゃまも対象。おじゃま自体は色数・消去個数・得点に算入しない。
既存一人用テンソルはおじゃまを表現しないためPvP学習には使用しない。

## 得点変換と全消し

各chain stepで `(step_score + attack_remainder) / garbage_rate` を攻撃量にする。剰余をプレイヤーごとに保持。累積scoreは表示・デバッグ用。
連鎖終了時に隠し行を含む盤面全体が空なら全消し予約を得る。次の通常色消去で+30を一度だけ発生させて消費する（基本得点が70未満でも発生）。その連鎖も全消しなら次回分を再獲得する。

## pending / confirmed と相殺

受け手のpendingは相手の進行中の連鎖からの予告。送信元の最終chain stepが終わるまで落下しない。
自分の攻撃は先に自分のconfirmed、その次にpendingを減らす。その後双方の余剰攻撃をmin分だけ相互相殺し、残量だけ相手へ送る。これにより処理順の有利不利を生まない。
連鎖終了Tickの攻撃も相殺してから残りを確定する。confirmedはChaining中には落下せず、各段で相殺可能。

## おじゃま落下

Tick開始時にReady、garbage_drop_due=true、confirmed>0なら、そのTickの行動はおじゃま落下でAction不要。新たにこのTickで確定した分は最速でも次Tickに落ちる。
1回最大30個。6個単位で全列に1個ずつ、端数は決定論的シャッフルした6列の先頭から重複なしで選ぶ。seed、tick、player_id、garbage_drop_countを64bit整数ミキサーへ入力する。外部RNGなし。
落下後はgarbage_drop_due=falseで配置機会を保証し、その配置でtrueに戻す。残り60個なら30落下→1ツモ→次回30落下。発火した場合は連鎖終了まで次回落下を延期する。
非死亡列が満杯でも、それだけではDeadにしない。まず従来のシャッフル順に従う各列への配分を可能な分だけ配置する。満杯列に割り当てられた分は、その同じシャッフル順を繰り返し走査し、空き列ごとに1個ずつ追加して再配分する。これにより空き列がある限り消失させず、余剰が特定列へ集中することも避ける。再配分を含め実落下は1回最大30個。
全列に空きがなくなったら配置を止め、配置できなかった分はconfirmedに残す。confirmedから減らす数とイベントのdroppedは実際に盤面へ入った数だけ。上書き・panicは行わず、落下終了後の `Board::is_game_over()` だけで死亡を判定する。
同Tickに別のプレイヤーが攻撃しても、開始時に予定済みの落下はそのまま実行する。今回の離散時間モデルとして固定した境界規則。

## 死亡と勝敗

安定した盤面（無連鎖配置後、最終連鎖段後、おじゃま落下後）に既存 `Board::is_game_over()` を使う。6列盤面のspawn_col=2、下から12段目が埋まる高さ12以上で死亡。連鎖途中では判定しない。非死亡列のoverflow自体は敗北条件にしない。再配分で死亡位置が埋まった場合は通常どおり死亡する。
片方のみ死亡なら相手勝利、双方同一Tick死亡はDraw。Deadは行動しない。通常APIから死亡が発生したTickで試合終了する。

## 実機との差と今後

満杯列から空き列への再配分と未配置分のconfirmed保持は、厳密な実機Tsuのoverflowフレーム挙動ではなく、研究用Tickモデルの決定論的近似。二回転も入力フレームではなく最終的な到達可能Placementとして近似する。
実時間、落下速度、ちぎり、操作速度、マージンタイム、実機の全フレーム挙動は扱わない。同時イベント、相殺優先順、全消しボーナス消費タイミング、落下Tickは上記の研究モデル。
既存Boardに合わせ通常色の連結判定は表示12行だけ。下側の隠し行は重力で落ちるが最上段の隠し行は重力対象外。この既存仕様を一人用とともに継承しており、完全な実機互換を主張しない。
将来はPvP用に相手盤面、おじゃま、phase、ツモindex、全消し予約などを観測へ追加し、同時Action/待機Tickを扱うMCTS adapterとself-playを作る。終端Valueはoutcome_for、scoreベースの既存rewardは使用しない。NN/MCTS/trainer/GUIのPvP化は今回の対象外。

## 検証

`cargo test -p puyo-core` は既存テスト、得点変換・対称相殺のunit test、`tests/pvp_tick.rs` の盤面ベースのintegration testを実行する。

`cargo run --release -p puyo-core --example pvp_smoke` はseed 0〜99の100試合。最大2000Tick、上限到達時はharness内のみDraw扱い。各試合の勝敗・Tick・配置数・最大連鎖、集計の攻撃・相殺・落下・相手連鎖中の配置数を出力する。攻撃/相殺/落下/連鎖中の複数配置の発生をassertする。

修正回帰テストは非死亡列満杯時の生存、再配分の保存・公平な分布・決定論性・30個上限、全列容量不足、Tsuのみの二回転と実配置、到達性・高さ制約を検証する。smokeは通常100試合に加え、100seedの非死亡列overflow fixtureを実行し、30個の再配分・生存・clone再現性をassertする。
