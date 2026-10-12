import type { Page } from "@playwright/test";

import { expect, test } from "../fixtures/base";

// LCP and layout shift on the built bundle (#67). Both regressions are
// invisible to a component test: jsdom has no geometry and no paint timing.
//
// Static pass only — the dev server serves an empty #root, so there is no
// static paint for the app to be compared against.
test.skip(!process.env.E2E_STATIC, "static-build pass only (E2E_STATIC=1)");

const VIEWPORTS = {
  phone: { width: 412, height: 823 },
  desktop: { width: 1350, height: 940 },
};

/**
 * LCP's own size rule, without its timing: the largest element holding text,
 * clipped to the viewport. Paint timing in headless Chromium is not reliable
 * enough to assert on (it sometimes holds the first frame until the scripts
 * have run), but the size comparison is what decides which paint is LCP.
 */
function largestTextBlock(page: Page) {
  return page.evaluate(() => {
    let best = { area: 0, text: "" };
    for (const el of document.querySelectorAll("#root *")) {
      const hasText = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent!.trim());
      if (!hasText) continue;
      const r = el.getBoundingClientRect();
      const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
      if (w * h > best.area) best = { area: w * h, text: el.textContent!.trim().slice(0, 40) };
    }
    return best;
  });
}

/** Sum every layout shift from the first byte, as CLS does on a short visit. */
async function observeShifts(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as { __cls: number };
    w.__cls = 0;
    new PerformanceObserver((list) => {
      for (const e of list.getEntries() as (PerformanceEntry & { value: number; hadRecentInput: boolean })[]) {
        if (!e.hadRecentInput) w.__cls += e.value;
      }
    }).observe({ type: "layout-shift", buffered: true });
  });
}

async function delay(page: Page, pattern: RegExp, ms: number) {
  await page.route(pattern, async (route) => {
    await new Promise((r) => setTimeout(r, ms));
    await route.continue();
  });
}

for (const [name, viewport] of Object.entries(VIEWPORTS)) {
  test.describe(`/home on a ${name}`, () => {
    test.use({ viewport });

    // The static HTML paints at first paint; the app replaces it once the
    // entry chunk has run — 1.8 s later on a throttled phone. If the app
    // paints anything larger, that becomes LCP and LCP waits for the chunk.
    test("the static paint is larger than anything the app paints", async ({ browser, page }) => {
      const noJs = await browser.newContext({ viewport, javaScriptEnabled: false, ignoreHTTPSErrors: true });
      const staticPage = await noJs.newPage();
      await staticPage.goto("/home");
      const before = await largestTextBlock(staticPage);
      await noJs.close();

      await page.goto("/home");
      await expect(page.getByText("GPU Capabilities")).toBeVisible();
      await expect(page.getByTestId("gpu-checking")).toHaveCount(0);
      const after = await largestTextBlock(page);

      expect(before.text).toMatch(/^Run machine-learning models/);
      expect(before.area, `the app paints "${after.text}" larger`).toBeGreaterThanOrEqual(after.area);
    });

    // The entry chunk is held so the static HTML is replaced mid-visit, and
    // the font longer still so it swaps in after the mount — the worst order
    // for a shift. The GPU check answering is the third mover.
    test("nothing shifts when the app mounts or the web font swaps in", async ({ page }) => {
      await observeShifts(page);
      await delay(page, /\/assets\/index-[^/]+\.js$/, 500);
      await delay(page, /geist-latin-wght-normal-[^/]+\.woff2$/, 1500);
      await page.goto("/home");
      await expect(page.getByText("GPU Capabilities")).toBeVisible();
      await expect(page.getByTestId("gpu-checking")).toHaveCount(0);
      await page.waitForFunction(() => document.fonts.check("16px 'Geist Variable'"));
      await page.waitForTimeout(500);

      const { cls, painted } = await page.evaluate(() => ({
        cls: (window as unknown as { __cls: number }).__cls,
        painted: performance.getEntriesByType("paint").length > 0,
      }));
      // An observer that never saw a frame would report 0 and pass.
      expect(painted).toBe(true);
      // "Good" is under 0.1 and the release measured 0.086; this pins half.
      // Measured with this order: 0.03 on a phone, 0.001 on a desktop.
      expect(cls).toBeLessThan(0.05);
    });
  });
}
