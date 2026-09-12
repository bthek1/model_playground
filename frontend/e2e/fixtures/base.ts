import { test as base, expect, type Page } from "@playwright/test";

import { installMockApi, seedTokens, type MockApiOptions } from "./mockApi";
import { probeWebGPU, type WebGPUStatus } from "./webgpu";

interface Fixtures {
  /** Install the stubbed Django API for this test. */
  mockApi: (options?: MockApiOptions) => Promise<void>;
  /** Boot the app with JWTs already in localStorage. */
  signedIn: () => Promise<void>;
  /** Real WebGPU status of this browser — use to skip GPU-only assertions. */
  webgpuStatus: WebGPUStatus;
}

/**
 * The Vite dev server occasionally serves a page whose module graph never
 * executes: it triggers its own full reload (`main.tsx?t=…`) while
 * re-optimising dependencies, and the reloaded document can arrive before the
 * optimiser has finished. The result is a served-but-empty `#root` that never
 * recovers on its own — a reload always fixes it.
 *
 * This is a dev-server characteristic, not an app bug (a production build has
 * no optimiser), but it makes the first navigation in a worker flaky. Every
 * `page.goto` therefore checks that the app actually mounted and reloads if it
 * didn't. Without this, roughly one navigation in four fails outright.
 */
const MOUNT_TIMEOUT_MS = 8_000;
const MAX_RELOADS = 3;

async function ensureMounted(page: Page): Promise<void> {
  const root = page.locator("#root > *").first();
  for (let attempt = 0; attempt <= MAX_RELOADS; attempt++) {
    try {
      await root.waitFor({ state: "attached", timeout: MOUNT_TIMEOUT_MS });
      return;
    } catch {
      if (attempt === MAX_RELOADS) {
        throw new Error(
          `App never mounted at ${page.url()} after ${MAX_RELOADS} reloads. ` +
            `Is the Vite dev server healthy?`,
        );
      }
      await page.reload();
    }
  }
}

/**
 * Every spec imports `test` and `expect` from here rather than from
 * @playwright/test directly. That gives one seam for adding fixtures — and
 * workarounds like ensureMounted — without touching every spec file.
 */
export const test = base.extend<Fixtures>({
  page: async ({ page }, use) => {
    const originalGoto = page.goto.bind(page);
    page.goto = async (url, options) => {
      const response = await originalGoto(url, options);
      // Only app routes mount React; about:blank and friends are left alone.
      if (page.url().startsWith("http")) await ensureMounted(page);
      return response;
    };
    await use(page);
  },

  mockApi: async ({ page }, use) => {
    await use((options) => installMockApi(page, options));
  },

  signedIn: async ({ page }, use) => {
    await use(() => seedTokens(page));
  },

  webgpuStatus: async ({ page }, use) => {
    // Must be probed from a **served origin**, not about:blank.
    //
    // `navigator.gpu` is only exposed in a secure context, and about:blank has
    // an opaque origin that does not qualify — so probing there reports
    // "unsupported" on every machine, working GPU or not, and every spec in
    // e2e/specs/webgpu/ skips itself for a reason that is not true. It is the
    // same secure-context trap the app itself documents for LAN origins
    // (docs/explanations/webgpu-inference.md); the test harness had it too.
    //
    // An app route rather than a static asset, because `page.goto` below is
    // wrapped in `ensureMounted` and a bare asset has no app to mount.
    await page.goto("/");
    await use(await probeWebGPU(page));
  },
});

export { expect };
