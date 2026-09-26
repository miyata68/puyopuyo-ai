import type { WasmGame } from "./types";

const DEFAULT_MODEL_NAME = "puyo_model.bin";

function getModelUrl(modelName: string): string {
  return `/models/${encodeURIComponent(modelName)}`;
}

export async function loadNnModel(
  game: WasmGame,
  modelName: string = DEFAULT_MODEL_NAME
): Promise<boolean> {
  const modelUrl = getModelUrl(modelName);

  try {
    const modelResponse = await fetch(modelUrl, {
      cache: "no-store",
    });

    if (!modelResponse.ok) {
      console.warn(
        `NN model file not found: ${modelName}. Using heuristic AI.`
      );
      return false;
    }

    const modelBytes = new Uint8Array(
      await modelResponse.arrayBuffer()
    );

    const success = game.load_nn_model(modelBytes);

    if (!success) {
      console.warn(
        `NN model is incompatible with current architecture: ${modelName}. Using heuristic AI.`
      );
      return false;
    }

    console.log(`NN policy model loaded: ${modelName}`);
    return true;
  } catch (e) {
    console.warn(
      `Failed to load NN model: ${modelName}`,
      e
    );
    return false;
  }
}

export async function loadNnModelWithMcts(
  game: WasmGame,
  numSimulations: number,
  modelName: string = DEFAULT_MODEL_NAME
): Promise<boolean> {
  const modelUrl = getModelUrl(modelName);

  try {
    const modelResponse = await fetch(modelUrl, {
      cache: "no-store",
    });

    if (!modelResponse.ok) {
      console.warn(
        `NN model file not found: ${modelName}. Using heuristic AI.`
      );
      return false;
    }

    const modelBytes = new Uint8Array(
      await modelResponse.arrayBuffer()
    );

    const success = game.load_nn_model_with_mcts(
      modelBytes,
      numSimulations
    );

    if (!success) {
      console.warn(
        `NN model is incompatible with current architecture: ${modelName}. Using heuristic AI.`
      );
      return false;
    }

    console.log(
      `NN MCTS model loaded: ${modelName} (simulations=${numSimulations})`
    );

    return true;
  } catch (e) {
    console.warn(
      `Failed to load NN MCTS model: ${modelName}`,
      e
    );
    return false;
  }
}