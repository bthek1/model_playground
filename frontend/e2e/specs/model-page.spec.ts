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
