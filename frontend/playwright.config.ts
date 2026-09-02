import { defineConfig, devices } from "@playwright/test";

// End-to-end tests. These complement — never replace — the Vitest + MSW
// component tests under `src/`. See docs/guides/e2e-testing.md.
//
// Two repo-specific facts drive most of this config:
//
//  1. The dev server is HTTPS with a self-signed cert (@vitejs/plugin-basic-ssl),
//     because `navigator.gpu` is only exposed in a secure context. So every
//     browser context AND the webServer health check need `ignoreHTTPSErrors`.
//  2. WebGPU availability depends on the machine's GPU and driver, so GPU specs
//     live in their own project and self-skip when no device can be acquired.

const BASE_URL = process.env.E2E_BASE_URL ?? "https://localhost:5180";

// `@backend` specs need a running Django + Postgres and a seeded user. They are
// excluded by default so `just fe-e2e` works with nothing but Node installed.
// `just fe-e2e-full` sets E2E_BACKEND=1 to opt in.
const useBackend = !!process.env.E2E_BACKEND;

// `@slow` specs reach Hugging Face and download real ONNX weights (80-220 MB a
// model), so they are excluded by default too — they turn a ~20s suite into a
// multi-minute one and they fail on an offline machine. `just fe-e2e-slow` sets
// E2E_SLOW=1 to opt in. They are the only tests that exercise a real ONNX
// Runtime session, which is where the audio bugs of 2026-09-01 lived.
const useSlow = !!process.env.E2E_SLOW;

const excludedTags = [
  useBackend ? null : "@backend",
  useSlow ? null : "@slow",
].filter(Boolean);

export default defineConfig({
  testDir: "./e2e/specs",
  outputDir: "./test-results",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  // Model fetches and GPU warm-up are slow; the default 30s is too tight.
  // `@slow` specs override this per-test — a 220 MB download needs minutes.
  timeout: 60_000,
  expect: { timeout: 10_000 },

  grepInvert: excludedTags.length
    ? new RegExp(excludedTags.join("|"))
    : undefined,

  reporter: process.env.CI
    ? [["github"], ["html", { open: "never" }], ["list"]]
    : [["html", { open: "never" }], ["list"]],

  use: {
    baseURL: BASE_URL,
    // Required: the dev server's cert is self-signed (see note 1 above).
    ignoreHTTPSErrors: true,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },

  projects: [
    // Logs in once through the real UI and saves storage state for the
    // authenticated specs. Only registered when a backend is actually running:
    // `grepInvert` filters by test *title*, and this one isn't @backend-tagged,
    // so leaving it in the list unconditionally makes the default (no-backend)
    // run fail on a login it can never complete.
    ...(useBackend
      ? [
          {
            name: "setup",
            testMatch: /global\.setup\.ts$/,
            use: { ...devices["Desktop Chrome"] },
          },
        ]
      : []),

    // Main functional suite.
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
      testIgnore: /webgpu\//,
      dependencies: useBackend ? ["setup"] : [],
    },

    // Cross-browser smoke subset only — full duplication buys little and
    // doubles the runtime.
    {
      name: "firefox",
      grep: /@smoke/,
      testIgnore: /webgpu\//,
      use: {
        ...devices["Desktop Firefox"],
        // Firefox on Linux/macOS still hides WebGPU behind a flag.
        launchOptions: { firefoxUserPrefs: { "dom.webgpu.enabled": true } },
      },
    },

    // WebGPU specs. Kept separate because they need a real GPU; they skip
    // themselves when detectWebGPU() does not report "ready".
    {
      name: "webgpu",
      testMatch: /webgpu\/.*\.spec\.ts/,
      use: {
        ...devices["Desktop Chrome"],
        // Playwright's default headless build ships no WebGPU at all, so these
        // specs need real Chrome/Chromium. Even then a machine with no GPU
        // device node (no /dev/dri) reports "unsupported" and the specs skip.
        channel: "chromium",
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      },
    },
  ],

  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5180",
    url: BASE_URL,
    // Without this the health check can't fetch the self-signed URL and
    // Playwright hangs until timeout against a server it can already reach.
    ignoreHTTPSErrors: true,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
