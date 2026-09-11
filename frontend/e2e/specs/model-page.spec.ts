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
  { path: "/vad", heading: "Voice Activity Detection", downloads: true },
  {
    path: "/image-classification",
    heading: "Image Classification",
    downloads: true,
  },
  { path: "/depth", heading: "Depth Estimation", downloads: true },
  { path: "/object-detection", heading: "Object Detection", downloads: true },
  {
    path: "/segmentation",
    heading: "Image Segmentation",
    downloads: true,
  },
  {
    path: "/zero-shot-image-classification",
    heading: "Zero-Shot Image Classification",
    downloads: true,
  },
  {
    path: "/zero-shot-object-detection",
    heading: "Zero-Shot Object Detection",
    downloads: true,
  },
  {
    path: "/image-features",
    heading: "Image Feature Extraction",
    downloads: true,
  },
  { path: "/mask-generation", heading: "Mask Generation", downloads: true },
  { path: "/image-to-text", heading: "Image to Text", downloads: true },
  { path: "/pose", heading: "Keypoint Detection", downloads: true },
  {
    path: "/video-classification",
    heading: "Video Classification",
    downloads: true,
  },
  {
    path: "/background-removal",
    heading: "Background Removal",
    downloads: true,
  },
  { path: "/super-resolution", heading: "Super Resolution", downloads: true },
  // Both runtimes on one page: the depth model downloads *and* a GPU is
  // probed. It is still a downloading route — the probe is free and additional.
  { path: "/image-to-3d", heading: "Image to 3D", downloads: true },
  // Compile-only: no weights, so LOAD auto-runs as a device probe (§7).
  { path: "/tensor", heading: "Tensor Arithmetic", downloads: false },
  // A task with nothing behind it renders the same page with empty slots, so an
  // unimplemented task reads as this page without a model — not another app.
  { path: "/tasks/text-to-image", heading: "Text to Image", downloads: false },
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
// Surviving a refresh (model/useModelSelection.ts). A Worker cannot outlive a
// page load, so what is restored is the *selection* and — only when the weights
// are already in Cache Storage — the load itself. The distinction is the whole
// guardrail: consent to spend bandwidth is never inferred, but re-reading bytes
// already on the machine spends none.
// ---------------------------------------------------------------------------

const KOKORO = "onnx-community/Kokoro-82M-v1.0-ONNX";

test.describe("a model page after a refresh", () => {
  test("restores the selected model without downloading anything", async ({
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
    await page.goto("/text-to-speech");
    await model.button(/MMS English/).click();

    await page.reload();

    // The choice survived…
    await expect(model.button(/MMS English/)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    // …and, with nothing cached, the page still asks before spending anything.
    await expect(model.button(/^Load model$/)).toBeVisible();
    expect(hubRequests, "a refresh re-downloaded weights").toEqual([]);
  });

  test("never resumes a load, even for weights already on this machine", async ({
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
    await page.goto("/text-to-speech");

    // Seed what a previous session leaves behind: the selection, and a file in
    // the bucket Transformers.js caches into. An older build also stored an
    // `autoResume` consent flag here and turned it into a download on the next
    // visit; the flag is seeded too, so a stale blob in a real user's browser
    // cannot bring that behaviour back.
    await page.evaluate(
      async ([modelId]) => {
        localStorage.setItem(
          "model-prefs",
          JSON.stringify({
            state: { selected: { tts: modelId }, autoResume: { tts: true } },
            version: 0,
          }),
        );
        const cache = await caches.open("transformers-cache");
        await cache.put(
          `https://huggingface.co/${modelId}/resolve/main/onnx/model.onnx`,
          new Response("weights"),
        );
      },
      [KOKORO],
    );

    await page.reload();

    // The badge says the download would be free — and the page still waits to
    // be asked. Cheap is not the same as consented.
    await expect(model.cachedBadge(KOKORO)).toBeVisible();
    await expect(model.button(/^Load model \(cached\)$/)).toBeVisible();
    await expect(model.loadProgress).toBeHidden();
    expect(hubRequests, "a cached model loaded without being asked").toEqual([]);
  });

  test("does not resume a model whose weights are gone", async ({
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
    await page.goto("/text-to-speech");
    await page.evaluate(() =>
      localStorage.setItem(
        "model-prefs",
        JSON.stringify({
          state: { selected: {}, autoResume: { tts: true } },
          version: 0,
        }),
      ),
    );

    await page.reload();

    await expect(model.button(/^Load model$/)).toBeVisible();
    expect(hubRequests, "an uncached model resumed without asking").toEqual([]);
  });

  test("cancelling a download returns the page to idle", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    // Never answer: the load stays in flight until the user gives up on it.
    await page.route((url) => url.hostname.endsWith("huggingface.co"), () => {});

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");
    await model.button(/^Load model$/).click();

    await expect(model.loadProgress).toBeVisible();
    await model.cancelLoad.click();

    await expect(model.loadProgress).toBeHidden();
    await expect(model.button(/^Load model$/)).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// Layout. jsdom has no geometry, so the *arrangement* half of the pattern can
// only be asserted in a real browser: this is where "the output is beside the
// input, not below it" is actually checked (model-page-pattern.md §4).
// ---------------------------------------------------------------------------

const DESKTOP = { width: 1440, height: 900 };
const TABLET = { width: 1000, height: 800 };
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

  test("splits into a setup strip over two work columns at md", async ({
    page,
    mockApi,
  }) => {
    // The middle breakpoint: not yet room for a rail beside the workbench, but
    // plenty for INPUT and OUTPUT to share a row under it.
    await mockApi();
    await page.setViewportSize(TABLET);

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");
    await expect(model.emptyOutput).toBeVisible();

    const select = await model.slot(1).boundingBox();
    const { input, output } = await workbenchBoxes(model);
    expect(select, "the SELECT band was not laid out").not.toBeNull();

    // Setup sits entirely above the workbench...
    expect(select!.y + select!.height).toBeLessThanOrEqual(input.y + 1);
    // ...and the work bands are still a row, not a stack.
    expect(input.x + input.width).toBeLessThanOrEqual(output.x + 1);
    expect(Math.abs(input.y - output.y)).toBeLessThan(8);
  });

  test("keeps the transport under a short input, with no chasm", async ({
    page,
    mockApi,
  }) => {
    // The regression the sticky-vs-pinned choice guards: /asr's input surface is
    // three sample-clip buttons, so bottom-pinning Run left ~400px of nothing
    // between the clips and the control that acts on them.
    await mockApi();
    await page.setViewportSize(DESKTOP);

    const model = new ModelPageObject(page);
    await page.goto("/asr");
    await expect(model.emptyOutput).toBeVisible();

    const clips = await model.button(/^JFK$/).boundingBox();
    const start = await model.button(/Start listening/).boundingBox();
    expect(clips, "the sample clips were not laid out").not.toBeNull();
    expect(start, "the transport was not laid out").not.toBeNull();

    const gap = start!.y - (clips!.y + clips!.height);
    expect(gap, "the transport row floated away from the input").toBeLessThan(
      120,
    );
  });

  test("fills the output column and lets the rail hug its content", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.setViewportSize(DESKTOP);

    const model = new ModelPageObject(page);
    await page.goto("/text-to-speech");
    await expect(model.emptyOutput).toBeVisible();

    const band = await model.slot(4).boundingBox();
    const panel = await model.outputPanel.boundingBox();
    expect(band, "the OUTPUT band was not laid out").not.toBeNull();
    expect(panel, "the OUTPUT panel was not laid out").not.toBeNull();
    // The panel owns the column, so a landing result cannot resize the card.
    expect(panel!.height).toBeGreaterThan(band!.height * 0.8);

    // The rail, by contrast, hugs: a full-height bordered panel holding two
    // short bands is a large rectangle of nothing.
    const rail = await model.slot(1).evaluate(
      (el) => el.parentElement!.getBoundingClientRect().height,
    );
    expect(rail).toBeLessThan(band!.height);
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
