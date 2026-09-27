# 仕様書 目次

## プロジェクト概要

ぷよぷよの盤面シミュレーションと連鎖エンジンを備えたAI学習ツール。
AIが算出した「次の一手」を見て、連鎖の組み方を学習できる。

## 仕様書一覧

| ファイル | 概要 |
|----------|------|
| [01-architecture.md](./01-architecture.md) | システム構成・用語定義 |
| [02-board.md](./02-board.md) | 盤面・連鎖処理 |
| [03-piece.md](./03-piece.md) | ぷよ組・ツモ |
| [05-score.md](./05-score.md) | スコア計算 |
| [06-game.md](./06-game.md) | ゲーム進行 |
| [07-rng.md](./07-rng.md) | 乱数生成 |
| [08-ai-eval.md](./08-ai-eval.md) | AI 評価関数 |
| [09-ai-search.md](./09-ai-search.md) | AI 探索 |
| [10-wasm-bridge.md](./10-wasm-bridge.md) | WASM ブリッジ |
| [11-frontend.md](./11-frontend.md) | フロントエンド |
| [12-nn.md](./12-nn.md) | ニューラルネットワーク |
| [13-trainer.md](./13-trainer.md) | 訓練パイプライン |
| [14-pvp-tick-engine.md](./14-pvp-tick-engine.md) | 決定論的2人対戦Tickエンジン |
| [15-pvp-alphazero.md](./15-pvp-alphazero.md) | PvP NN・独立探索・勝敗教師・自己対局学習 |
| [16-pvp-training-evaluation.md](./16-pvp-training-evaluation.md) | 配置数温度・Replay Window・Validation・モデル間対戦評価 |
