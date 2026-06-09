import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

// Standalone Vitest config (does not load the Lovable/TanStack Vite plugins,
// which are unnecessary for unit-testing pure library functions).
export default defineConfig({
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
});
