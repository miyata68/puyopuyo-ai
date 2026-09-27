# PvP学習・世代間評価

## 温度スケジュール

SelfPlayConfigのtemperature_scheduleはTemperatureSchedule::PieceIndex(20)が既定。
各decisionのTick開始状態から、自分自身のpiece_index（配置済みツモ数）を読む。
0..19ではtemperature=1.0のvisit sampling、20以降はtemperature=0の最大visit argmax。
温度0の同点は小さいaction indexを選ぶ。NN入力やゲームルールは変えない。
同じTickでもP1のpiece_index=25、P2=10ならP1はargmax、P2はsampling。
連鎖・おじゃま落下だけでTickが進んでもpiece scheduleは進まない。

canonical CLIは `--temperature 1.0 --temperature-drop-piece 20`。
明示的な `--temperature-drop-tick 100` はLegacyTickとして旧動作を保ち、deprecated警告を出す。
pieceとtickの両オプションを明示するとエラー。drop-piece=0、またはtemperature=0なら最初からargmax。
旧Tick100は平均40～50Tickの対局では多くがsamplingのまま終わるため、配置数ベースに変更した。
起動時にscheduleと閾値、終了時にsampled_decisions/argmax_decisionsをP1/P2別で表示する。
診断のdecision数には打切り試合も含むが、その試合の教師sampleは従来通り全て破棄する。

## Replay Window

推奨3世代を `--data-paths "iter002.bin,iter003.bin,iter004.bin"` で明示する。
世代番号を自動判定したりファイルを削除したりはしない。従来の単一 `--data` も使用可能で、両方の指定はエラー。
カンマで分割し前後空白を除去。空要素・重複パスを拒否する。
実パスをcanonicalizeして相対パス・シンボリックリンクの別名も検出し、Windowsでは区切りを `/`、文字を小文字に正規化する。
全ファイルをPvpAlphaZeroDataset::loadで読み、load内のvalidateにより全metadataと全sampleを検証する。
format_version/model_kind/board_channels/rows/cols/context_size/num_actionsの不一致はエラー。
保存形式v1は変更せず既存datasetを利用できる。

全datasetのsampleを結合後、Train partitionのunionから一様に復元抽出する。
sample数の多い世代はその分多く選ばれる。世代別equal weightingはしない。
乱数は学習seedで初期化し、index選択には剰余biasを避けるrejection samplingを使う。
色置換は選ばれたTrain sampleのコピーだけに適用。元のTrain/Val poolは変更しない。

## 固定train/validation split

既定 `--validation-fraction 0.10 --validation-seed 42`。
有限の `0 < fraction < 0.5` のみ許す。TrainまたはValが空ならエラー。
分割はoptimizer作成・step 0評価より前に行う。

正規化した絶対sourceパスUTF-8、0byte delimiter、sample indexのu64 little-endian、validation_seedのu64 little-endianをFNV-1aに入力し、SplitMix64 finalizerを適用する。
hashの上位53bitを2^53で割った値がfraction未満ならVal、それ以外はTrain。
DefaultHasher・thread_rng・時刻・pool全体のshuffleには依存しない。
同じsource/index/seed/fractionなら、Replay Windowの追加・削除・順序変更でも所属は変わらない。
datasetを移動・改名したり内部sample順序を変えたりすると所属は変わる。実測割合は確率的に指定値へ近づき、件数を厳密に10%へ調整しない。

ログは各sourceのsamples/train/valと、total_train/total_val/validation_fraction_actualを表示する。
データはsample単位分割であり、同一試合の別局面が両partitionに入る可能性がある。
未知sampleへの誤差を測るもので、未知試合・世代・対戦相手への一般化や強さの保証ではない。
別パスの内容コピーやhard linkまで同一datasetとして検出するわけではないため、同じdatasetを別名で重複投入しないこと。

## Policy/Value評価

teacherはroot visitsを正規化したp、modelは合法手だけでsoftmaxしたq。

| 指標 | 定義 |
|---|---|
| policy_ce | `-sum(p * ln(q))` |
| target_entropy | `-sum(p * ln(p))`、p=0の項は0 |
| policy_kl | `max(0, policy_ce - target_entropy)` |
| value_mse | 平均 `(raw_value - 最終勝敗z)^2` |

自然対数を使用。KLの負の微小誤差は0に丸める。
softmax分母に入るのはlegal_mask=trueの全Action。教師0の合法手も含め、illegalだけを除く。
評価ではf64のlog-sum-expでCE/Entropyを計算し、最後の端数batchもsample数で加重平均する。
Value教師はwin +1 / draw 0 / loss -1のままでscore/chain/garbageやsolo value transformは使わない。

既定 `--eval-interval 100 --eval-samples 4096 --eval-batch-size 512`。
Train/Valごとに最大4096件を一度だけ固定抽出し、同じrun中は同じsubsetを毎回測る。
subset用乱数は学習のbatch/augmentation乱数と別で、validation_seedとsplit用定数から導出する。
両subsetとも色置換しない。step 0、interval毎、最終stepに以下を出す。

```text
EVAL step=0 split=train n=4096 policy_ce=... target_entropy=... policy_kl=... value_mse=...
EVAL step=0 split=val n=4096 policy_ce=... target_entropy=... policy_kl=... value_mse=...
TRAIN step=1 policy_ce=... value_mse=... total=...
FINAL_VALIDATION n=... policy_ce=... target_entropy=... policy_kl=... value_mse=...
```

最後はVal全件をeval_batch_size以下に分けて評価する。全件を1個の巨大Tensorにはしない。
Replay poolはCPUメモリへ全件ロードするので、巨大datasetではホストメモリの容量が制約になる。

Burn 0.19.1ではBatchNorm::forwardがbackendのad_enabledでtrain/inferenceを分岐する。
evaluate_snapshotはAutodiffModule::valid()でinner backendのsnapshotへ変換し、inference forwardのみ行う。
optimizerを参照せず、backwardを呼ばず、BatchNormの学習forwardを使わない。
テストは評価前後の全record（重みとBN state）の同一性とbatch分割平均を検証する。

## Model-vs-Model

`pvp-eval`はA/Bそれぞれのmetadataと重みをロードし、2つのInference Serverを作る。
未存在・不完全・soloモデルを拒否する。共通PvP入出力次元ならarchitectureサイズは異なってもよい。

PvpSearchV1::search_with_providersのSelf providerはpriorとleaf Valueに使用する。
Opponent providerは全ノードのcached baselineとforced Tick中の相手Actionに使用する。
Aの探索はself=A/opponent=B、Bはself=B/opponent=A。既存search APIは同一providerを両方に渡すwrapperとして残す。
相手baselineは各Tickの候補適用前に確定し、同じprepared nodeの全候補で共有する。
実対局も同一の不変rootから両者をsearchし、両Actionが揃ってからstepする。

評価の探索noiseは必ずfalse、Action温度は必ず0。CLIで上書きできない。
1つのmatch seedにつきA=P1/B=P2とB=P1/A=P2の2試合を実施する。
250pairsは500games。共通ツモseedで席を交換し、席効果を平均化する。
エンジンのおじゃま端数配置は席もseedに含むので、入替えで全試合の勝敗が完全反転する保証はない。

Model A/Bそれぞれwins/losses/draws/truncated/score/score_rate、A_as_p1/p2を表示。
scoreはwins+0.5*draws、score_rateの分母は**完了試合数**（wins+losses+draws）。
Ongoingのままmax_ticksに達した試合はtruncatedで、Drawにしない。完了0ならscore_rate=NA。
paired分類はA_wins_both/B_wins_both/split_1_1/pairs_containing_draw/pairs_containing_truncation。
片方drawで片方truncatedのpairは最後の2分類の両方に数える（排他的分類ではない）。
二者間score rateを主指標とし、global Eloやpromotion/gatingは実装しない。

同じmodel/seed範囲/search設定/backend/実行条件で再現する設計。
CPU単一threadの小型モデルとmockで再現を確認する。CUDA並列動的batch・別GPUの浮動小数点差によるbitwise同一は保証しない。

## Model 4を作る順序（PowerShell）

以下は既存model_003/002とiter_002/003を利用する本運用用。既存の出力model_004/iter_004を上書きしたくない場合は、出力パスを変更する。
推奨値は500games、64simulations、64ch×6blocks、1000steps、batch128、LR0.001、Value重み1、max_ticks2000のまま。
実装検証ではこの大規模CUDAジョブは自動実行しない。

### A. Model 3 vs Model 2

```powershell
Set-Location C:\work\puyopuyo\puyopuyo-ai
cargo run --release -p puyo-trainer --bin pvp-eval -- `
  --model-a artifacts/pvp/model_003 `
  --model-b artifacts/pvp/model_002 `
  --pairs 250 --simulations 64 --seed-offset 2000000 `
  --max-ticks 2000 --backend cuda --threads 64 --inference-batch-size 128
```

### B. Iteration 4 self-play

```powershell
cargo run --release -p puyo-trainer --bin pvp-self-play -- `
  --games 500 --simulations 64 --model-path artifacts/pvp/model_003 `
  --output data/pvp/iter_004.bin --seed-offset 1001500 --max-ticks 2000 `
  --temperature 1.0 --temperature-drop-piece 20 --root-noise true `
  --swap-streams true --backend cuda --threads 64 --inference-batch-size 128
```

Iteration 1/2/3/4のstream割当はfalse/true/false/trueと交互にする。

### C. Replay Window 3でModel 4学習

```powershell
cargo run --release -p puyo-trainer --bin train-pvp -- `
  --data-paths "data/pvp/iter_002.bin,data/pvp/iter_003.bin,data/pvp/iter_004.bin" `
  --model-path artifacts/pvp/model_003 --output-model artifacts/pvp/model_004 `
  --steps 1000 --batch-size 128 --learning-rate 0.001 --value-loss-weight 1.0 `
  --validation-fraction 0.10 --validation-seed 42 `
  --eval-interval 100 --eval-samples 4096 --eval-batch-size 512 `
  --seed 42 --backend cuda
```

### D. Model 4 vs Model 3

```powershell
cargo run --release -p puyo-trainer --bin pvp-eval -- `
  --model-a artifacts/pvp/model_004 `
  --model-b artifacts/pvp/model_003 `
  --pairs 250 --simulations 64 --seed-offset 2000500 `
  --max-ticks 2000 --backend cuda --threads 64 --inference-batch-size 128
```

CPUで使う場合は各cargoコマンドに `--no-default-features` を付け、`--backend cpu --threads 1`（trainにはthreads不要）を指定する。
既存solo binaryはCUDA importを持つため、`--no-default-features`でpackage全binaryを一括buildする用途には対応しない。PvP binaryを `--bin` で明示する。
