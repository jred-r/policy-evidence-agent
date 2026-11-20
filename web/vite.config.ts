import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const outputDirectory =
  process.env.STATIC_OUTPUT_DIR ??
  fileURLToPath(new URL("../src/qa/interfaces/static", import.meta.url));

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/@assistant-ui/")) return "assistant-ui";
          if (id.includes("/node_modules/react")) return "react-vendor";
          return undefined;
        },
      },
    },
  },
});
