import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["**/*.test.ts"],
    exclude: ["node_modules/**"],
    // Pulumi's mocks are process-global (`pulumi.runtime.setMocks`), so each
    // stack's test file needs its own process.
    pool: "forks",
  },
});
