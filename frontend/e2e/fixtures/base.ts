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
    // Needs a document before navigator is reachable; about:blank is enough.
    await page.goto("about:blank");
    await use(await probeWebGPU(page));
  },
});

export { expect };
