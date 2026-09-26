import { loadWasm } from "./wasm";
import { Renderer } from "./renderer";
import { GameLoop } from "./game-loop";
import { UI } from "./ui";
import { loadNnModel, loadNnModelWithMcts } from "./model-loader";
import { CELL_SIZE } from "./constants";
import type { BoardConfig } from "./constants";

function setupAiModeToggle(gameLoop: GameLoop): void {
  const aiModeSelect =
    document.getElementById("ai-mode-select") as HTMLSelectElement;

  const aiModeStatus =
    document.getElementById("ai-mode-status") as HTMLElement;

  const mctsOptions =
    document.getElementById("mcts-options") as HTMLElement;

  const mctsSimInput =
    document.getElementById("mcts-simulations") as HTMLInputElement;

  const modelSelect =
    document.getElementById("ai-model-select") as HTMLSelectElement;

  const refreshModelButton =
    document.getElementById(
      "refresh-model-list"
    ) as HTMLButtonElement;


  const getSelectedModel = (): string => {
    return modelSelect.value || "puyo_model.bin";
  };


  /**
   * public/models 以下のモデル一覧を取得する
   */
  const loadModelList = async (): Promise<void> => {
    const previousModel = modelSelect.value;

    refreshModelButton.disabled = true;

    try {
      const response = await fetch("/api/models", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(
          `Failed to load model list: ${response.status}`
        );
      }

      const models = (await response.json()) as string[];

      modelSelect.innerHTML = "";

      if (models.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "モデルなし";

        modelSelect.appendChild(option);
        modelSelect.disabled = true;

        aiModeStatus.textContent =
          "models フォルダに .bin がありません";

        return;
      }

      modelSelect.disabled = false;

      for (const model of models) {
        const option = document.createElement("option");

        option.value = model;
        option.textContent = model;

        modelSelect.appendChild(option);
      }

      // 更新前に選択していたモデルがまだ存在すれば維持する
      if (
        previousModel &&
        models.includes(previousModel)
      ) {
        modelSelect.value = previousModel;
      }
      // puyo_model.bin があればデフォルトにする
      else if (models.includes("puyo_model.bin")) {
        modelSelect.value = "puyo_model.bin";
      }
      // なければ先頭
      else {
        modelSelect.value = models[0];
      }
    } catch (e) {
      console.error("Failed to load model list:", e);

      aiModeStatus.textContent =
        "モデル一覧の取得に失敗しました";
    } finally {
      refreshModelButton.disabled = false;
    }
  };


  /**
   * 現在選択されているAIモードとモデルを適用する
   */
  const applyAiMode = async (
    game: ReturnType<typeof gameLoop.getGame>
  ) => {
    const mode = aiModeSelect.value;
    const modelName = getSelectedModel();

    if (mode === "nn") {
      aiModeStatus.textContent =
        `モデル読み込み中: ${modelName}`;

      const success = await loadNnModel(
        game,
        modelName
      );

      if (success) {
        aiModeStatus.textContent =
          `NN AI 有効: ${modelName}`;
      } else {
        aiModeStatus.textContent =
          `モデル読み込み失敗: ${modelName}`;

        aiModeSelect.value = "heuristic";
        game.use_heuristic();
      }

    } else if (mode === "nn-mcts") {
      const numSim =
        parseInt(mctsSimInput.value, 10) || 50;

      aiModeStatus.textContent =
        `MCTS読み込み中: ${modelName} (${numSim}sim)`;

      const success =
        await loadNnModelWithMcts(
          game,
          numSim,
          modelName
        );

      if (success) {
        aiModeStatus.textContent =
          `NN MCTS 有効: ${modelName} (${numSim}sim)`;
      } else {
        aiModeStatus.textContent =
          `モデル読み込み失敗: ${modelName}`;

        aiModeSelect.value = "heuristic";
        mctsOptions.style.display = "none";
        game.use_heuristic();
      }

    } else {
      game.use_heuristic();
      aiModeStatus.textContent = "";
    }
  };


  /**
   * AIモード変更
   */
  aiModeSelect.addEventListener(
    "change",
    async () => {
      const mode = aiModeSelect.value;

      mctsOptions.style.display =
        mode === "nn-mcts" ? "" : "none";

      await applyAiMode(
        gameLoop.getGame()
      );
    }
  );


  /**
   * モデル変更
   *
   * NN / NN+MCTS 使用中なら、
   * 選択したモデルを即座にロードする。
   */
  modelSelect.addEventListener(
    "change",
    async () => {
      if (
        aiModeSelect.value === "nn" ||
        aiModeSelect.value === "nn-mcts"
      ) {
        await applyAiMode(
          gameLoop.getGame()
        );
      }
    }
  );


  /**
   * モデル一覧更新
   *
   * AlphaZero学習中に新しい .bin が生成された場合も
   * ページリロードなしで一覧を更新できる。
   */
  refreshModelButton.addEventListener(
    "click",
    async () => {
      await loadModelList();

      if (
        aiModeSelect.value === "nn" ||
        aiModeSelect.value === "nn-mcts"
      ) {
        await applyAiMode(
          gameLoop.getGame()
        );
      }
    }
  );


  /**
   * MCTS simulation数変更
   */
  mctsSimInput.addEventListener(
    "change",
    () => {
      if (
        aiModeSelect.value === "nn-mcts"
      ) {
        const numSim =
          parseInt(
            mctsSimInput.value,
            10
          ) || 50;

        gameLoop
          .getGame()
          .set_mcts_simulations(numSim);

        aiModeStatus.textContent =
          `NN MCTS 有効: ${getSelectedModel()} (${numSim}sim)`;
      }
    }
  );


  /**
   * Rキー等によるリスタート時にも
   * 現在選択されているモデルを再ロードする
   */
  gameLoop.setOnRestart(applyAiMode);


  /**
   * 初期モデル一覧取得
   */
  void loadModelList();
}

async function main() {
  const wasm = await loadWasm();

  const COLS = 6;
  const ROWS = 14;
  const NUM_COLORS = 4;

  const getTsumoSeed = (): number => {
    const input =
      document.getElementById("tsumo-seed") as HTMLInputElement;

    const parsed = Number.parseInt(input.value, 10);

    if (!Number.isFinite(parsed)) {
      return 123456;
    }

    return Math.max(
      0,
      Math.min(4294967295, parsed)
    );
  };

  const createGame = () => {
    const game =
      new wasm.WasmGame(COLS, ROWS, NUM_COLORS);

    game.restart_with_seed(getTsumoSeed());

    return game;
  };


  // const createGame = () => new wasm.WasmGame(COLS, ROWS, NUM_COLORS);
  const game = createGame();

  const config: BoardConfig = {
    cols: game.board_cols(),
    rows: game.board_rows(),
    visibleRows: game.board_visible_rows(),
    numColors: game.num_colors(),
    boardWidth: game.board_cols() * CELL_SIZE,
    boardHeight: game.board_rows() * CELL_SIZE,
  };

  const boardCanvas = document.getElementById("board-canvas") as HTMLCanvasElement;
  const nextCanvas = document.getElementById("next-canvas") as HTMLCanvasElement;
  const nextNextCanvas = document.getElementById("next-next-canvas") as HTMLCanvasElement;

  boardCanvas.width = config.boardWidth;
  boardCanvas.height = config.boardHeight;

  const renderer = new Renderer(boardCanvas, nextCanvas, nextNextCanvas, config);
  const ui = new UI();

  const gameLoop = new GameLoop(game, renderer, ui, createGame);
  gameLoop.start();

  setupAiModeToggle(gameLoop);
}

main().catch(console.error);
