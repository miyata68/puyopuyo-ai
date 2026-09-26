import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import { readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const modelsDir = fileURLToPath(
  new URL("./public/models/", import.meta.url)
);

export default defineConfig({
  plugins: [
    wasm(),

    // public/models 以下の .bin 一覧を返すAPI
    {
      name: "model-list-api",

      configureServer(server) {
        server.middlewares.use("/api/models", async (_req, res) => {
          try {
            const entries = await readdir(modelsDir, {
              withFileTypes: true,
            });

            const models = entries
              .filter(
                (entry) =>
                  entry.isFile() &&
                  entry.name.toLowerCase().endsWith(".bin")
              )
              .map((entry) => entry.name)
              .sort((a, b) =>
                a.localeCompare(b, undefined, {
                  numeric: true,
                  sensitivity: "base",
                })
              );

            res.statusCode = 200;
            res.setHeader(
              "Content-Type",
              "application/json; charset=utf-8"
            );
            res.setHeader("Cache-Control", "no-store");
            res.end(JSON.stringify(models));
          } catch (e) {
            console.error("Failed to list models:", e);

            res.statusCode = 500;
            res.setHeader(
              "Content-Type",
              "application/json; charset=utf-8"
            );
            res.end(JSON.stringify([]));
          }
        });
      },
    },
  ],

  build: {
    target: "esnext",
  },

  server: {
    host: true,
    fs: {
      allow: [".", "wasm-pkg"],
    },
  },
});