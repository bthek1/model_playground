import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// The four-slot contract, asserted once across every real task route
// (docs/standards/model-page-pattern.md).
//
// The per-route specs cover what each task *does*; this one covers what they all
// promise regardless of modality: four bands in order, an OUTPUT that is present
// before there is anything to put in it, and — for the routes that download
// weights — nothing fetched until the user asks.
//
// It exists because the five audio routes drifted apart precisely in this shape
// before the pattern was extracted: each had its own header, its own status
// line, and its own idea of when a download should start.

/** Every route that renders the shell, and whether LOAD is a download. */
const TASK_ROUTES = [
  { path: "/asr", heading: "Automatic Speech Recognition", downloads: true },
  {
    path: "/audio-classification",
    heading: "Audio Classification",
    downloads: true,
  },
  { path: "/text-to-speech", heading: "Text to Speech", downloads: true },
  { path: "/text-to-audio", heading: "Text to Audio", downloads: true },
  { path: "/audio-to-audio", heading: "Audio to Audio", downloads: true },
  // Compile-only: no weights, so LOAD auto-runs as a device probe (§7).
  { path: "/tensor", heading: "Tensor Arithmetic", downloads: false },
  // A task with nothing behind it renders the same page with empty slots, so an
  // unimplemented task reads as this page without a model — not another app.
  { path: "/tasks/depth-estimation", heading: "Depth Estimation", downloads: false },
] as const;

test.describe("the four-slot model page contract", () => {
  for (const route of TASK_ROUTES) {
    test(`${route.path} renders Select → Load → Run → Output`, async ({
      page,
      mockApi,
    }) => {
      await mockApi();
      // Block the Hub outright: this spec must never pull weights.
      await page.route(
        (url) => url.hostname.endsWith("huggingface.co"),
        (r) => r.abort(),
      );

      const model = new ModelPageObject(page);
      await page.goto(route.path);

      await expect(
        page.getByRole("heading", { name: route.heading, level: 1 }),
      ).toBeVisible();

      // Four bands, in order, none of them optional.
      await expect(model.slots).toHaveCount(4);
      for (const step of [1, 2, 3, 4] as const) {
        await expect(model.slot(step)).toBeVisible();
      }

      // OUTPUT exists before there is any output — the page must not grow a
      // section (and shove everything down) the moment a result lands.
      await expect(model.emptyOutput).toBeVisible();
    });

    if (route.downloads) {
      test(`${route.path} downloads nothing until the user asks`, async ({
        page,
        mockApi,
      }) => {
        await mockApi();
        const hubRequests: string[] = [];
        await page.route(
          (url) => url.hostname.endsWith("huggingface.co"),
          (r) => {
            hubRequests.push(r.request().url());
            return r.abort();
          },
        );

        const model = new ModelPageObject(page);
        await page.goto(route.path);
        await expect(
          page.getByRole("heading", { name: route.heading, level: 1 }),
        ).toBeVisible();

        // The LOAD slot is offering the action, which means it hasn't started.
        await expect(model.button(/^Load model$/)).toBeVisible();
        expect(
          hubRequests,
          `${route.path} fetched weights on navigation`,
        ).toEqual([]);
      });
    }
  }
});

// ---------------------------------------------------------------------------
// Layout. jsdom has no geometry, so the *arrangement* half of the pattern can
// only be asserted in a real browser: this is where "the output is beside the
// input, not below it" is actually checked (model-page-pattern.md §4).
// ---------------------------------------------------------------------------

const DESKTOP = { width: 1440, height: 900 };
const PHONE = { width: 375, height: 812 };

/** Bounding boxes of the INPUT and OUTPUT bands, both required to be laid out. */
async function workbenchBoxes(model: ModelPageObject) {
  const input = await model.slot(3).boundingBox();
  const output = await model.slot(4).boundingBox();
  if (!input || !output) throw new Error("a workbench band was not laid out");
  return { input, output };
}

test.describe("the model page arrangement", () => {
  test.beforeEach(async ({ page }) => {
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (r) => r.abort(),
    );
  });

  test("puts INPUT and OUTPUT side by side on a desktop viewport", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.setViewportSize(DESKTOP);

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");
    await expect(model.emptyOutput).toBeVisible();

    const { input, output } = await workbenchBoxes(model);
    // Horizontally disjoint...
    expect(
      input.x + input.width,
      "INPUT and OUTPUT overlap horizontally — they are not columns",
    ).toBeLessThanOrEqual(output.x + 1);
    // ...and vertically aligned, which is what makes them a row rather than a
    // stack that happens to be narrow.
    expect(Math.abs(input.y - output.y)).toBeLessThan(8);
  });

  test("keeps the result above the fold — the whole point of the change", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.setViewportSize(DESKTOP);

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");

    const box = await model.outputPanel.boundingBox();
    expect(box, "the OUTPUT panel was not laid out").not.toBeNull();
    expect(
      box!.y,
      "the OUTPUT panel starts below the fold on a 900px-tall viewport",
    ).toBeLessThan(DESKTOP.height);
  });

  test("never scrolls horizontally, at either viewport", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    for (const viewport of [DESKTOP, PHONE]) {
      await page.setViewportSize(viewport);
      await page.goto("/text-to-speech");
      await expect(model.emptyOutput).toBeVisible();

      // A grid child that refuses to shrink pushes the page wide; `min-w-0` on
      // both workbench columns is what prevents it.
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      expect(
        overflow,
        `horizontal overflow at ${viewport.width}px`,
      ).toBeLessThanOrEqual(1);
    }
  });

  test("stacks all four bands in pipeline order on a phone", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.setViewportSize(PHONE);

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");
    await expect(model.slots).toHaveCount(4);

    const tops: number[] = [];
    for (const step of [1, 2, 3, 4] as const) {
      const box = await model.slot(step).boundingBox();
      expect(box, `slot-${step} was not laid out`).not.toBeNull();
      tops.push(box!.y);
    }
    // Strictly increasing: one column, in pipeline order.
    for (let i = 1; i < tops.length; i += 1) {
      expect(tops[i], `slot-${i + 1} is not below slot-${i}`).toBeGreaterThan(
        tops[i - 1],
      );
    }
  });
});
