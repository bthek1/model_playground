import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// The mocked half of `/image-text-to-text`: routing, the four-slot shell, the
// gate, and the rules that say what is allowed to spend the user's bandwidth.
// No weights are downloaded here — the real model is `multimodal-models.spec.ts`.

test.describe("/image-text-to-text", () => {
  test("renders the four slots with an empty OUTPUT", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/image-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.outputPanel).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
  });

  // Every model on this page is past `LARGE_MODEL_BYTES` several times over, so
  // the gate is the spec: nothing is fetched until the user asks for it.
  test("downloads nothing until Load model is clicked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/image-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();

    // The cost is quoted — once, by the picker — before anything is fetched.
    await expect(model.sizeNote).toBeVisible();
    await expect(model.largeModelWarning).toBeVisible();
    expect(hubRequests, "no weights before the click").toEqual([]);

    // The input surface is alive and the trigger is dead: choosing is free,
    // generating is not.
    await expect(page.getByTestId("prompt-input")).toBeVisible();
    await expect(model.button(/^Generate$/)).toBeDisabled();

    // Pressing Load is what starts the download. With the Hub blocked it fails,
    // which is also the proof it was attempted.
    await model.load();
    await expect(model.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length, "load starts the download").toBeGreaterThan(0);
  });

  test("typing a question and picking a picture download nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/image-text-to-text");
    await page.getByTestId("prompt-input").fill("What colour is the sky?");
    await page
      .getByTestId("presets")
      .getByRole("button", { name: /how many people/i })
      .click();

    // Both are INPUT: they commit to nothing.
    expect(hubRequests, "INPUT never spends").toEqual([]);
    await expect(model.emptyOutput).toBeVisible();
  });

  test("the sidebar links to the route rather than the placeholder", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/image-text-to-text");
    // The taxonomy entry is wired, so the task no longer falls through to
    // `/tasks/$slug`.
    await expect(page).toHaveURL(/\/image-text-to-text$/);
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();
  });
});

// The second route on the same engine. It downloads nothing new — the catalogue
// and the worker are `/image-text-to-text`'s — so the only things worth asserting
// here are the shell, the gate, and the one rule this page is most able to break:
// the terse toggle looks like a filter, and must not behave like one.
test.describe("/visual-question-answering", () => {
  test("renders the four slots with an empty OUTPUT", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/visual-question-answering");
    await expect(
      page.getByRole("heading", { name: "Visual Question Answering" }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.outputPanel).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
  });

  test("downloads nothing until Load model is clicked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/visual-question-answering");
    await expect(
      page.getByRole("heading", { name: "Visual Question Answering" }),
    ).toBeVisible();

    await expect(model.sizeNote).toBeVisible();
    await expect(model.largeModelWarning).toBeVisible();
    expect(hubRequests, "no weights before the click").toEqual([]);

    await expect(model.button(/^Generate$/)).toBeDisabled();

    await model.load();
    await expect(model.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length, "load starts the download").toBeGreaterThan(0);
  });

  test("the terse toggle rewrites the prompt on screen and spends nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/visual-question-answering");
    await page.getByTestId("question-input").fill("What animal is this?");
    await expect(page.getByTestId("composed-prompt")).toHaveText(
      "What animal is this?",
    );

    // The whole demonstration, and the whole hazard: the prompt changes, the
    // page does not run, and nothing is fetched.
    await page.getByTestId("terse-toggle").check();
    await expect(page.getByTestId("composed-prompt")).toHaveText(
      "What animal is this? Answer in one word.",
    );
    expect(hubRequests, "INPUT never spends").toEqual([]);
    await expect(model.emptyOutput).toBeVisible();
  });

  test("the sidebar links to the route rather than the placeholder", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/visual-question-answering");
    await expect(page).toHaveURL(/\/visual-question-answering$/);
    await expect(
      page.getByRole("heading", { name: "Visual Question Answering" }),
    ).toBeVisible();
  });
});

// The third route on the same engine, and the one that needed a model rather
// than only a prompt. Mocked: the shell, the gate, and the two rules the frame
// controls are most able to break — changing the frame count and flipping the
// order are INPUT, and neither costs a decode or a download.
test.describe("/video-text-to-text", () => {
  test("renders the four slots with an empty OUTPUT", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/video-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Video Text to Text" }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.outputPanel).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
  });

  test("says it is a frame sampler in the header, before any result", async ({
    page,
    mockApi,
  }) => {
    // The same correctness requirement `/video-classification` carries: a page
    // called "Video Text to Text" that does not say what it does to the video
    // teaches something false about what the model understands.
    await mockApi();
    await new ModelPageObject(page).blockModelDownloads();
    await page.goto("/video-text-to-text");
    await expect(
      page.getByText(/frame sampler plus an image model/i),
    ).toBeVisible();
  });

  test("downloads nothing until Load model is clicked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/video-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Video Text to Text" }),
    ).toBeVisible();

    await expect(model.sizeNote).toBeVisible();
    expect(hubRequests, "no weights before the click").toEqual([]);

    // The frame controls are live while the trigger is dead: choosing is free.
    await page.getByTestId("frame-count").fill("8");
    await page.getByTestId("reverse-toggle").check();
    await page.getByTestId("question-input").fill("What happens here?");
    expect(hubRequests, "INPUT never spends").toEqual([]);
    await expect(model.button(/^Generate$/)).toBeDisabled();
    await expect(model.emptyOutput).toBeVisible();

    await model.load();
    await expect(model.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length, "load starts the download").toBeGreaterThan(0);
  });

  test("picking a clip fetches the preview lazily and decodes nothing", async ({
    page,
    mockApi,
  }) => {
    // `preload="none"` on the preview and no sampling until GENERATE: the clips
    // are megabytes, and browsing them must not pay for a decode.
    await mockApi();
    await new ModelPageObject(page).blockModelDownloads();

    const clipRequests: string[] = [];
    await page.route(
      (url) => url.pathname.endsWith(".mp4"),
      (route) => {
        clipRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/video-text-to-text");
    await page.getByRole("button", { name: "Interview" }).click();
    await expect(page.locator('video[aria-label*="Interview"]')).toBeVisible();
    expect(clipRequests, "the preview does not preload").toEqual([]);
  });

  test("the sidebar links to the route rather than the placeholder", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/video-text-to-text");
    await expect(page).toHaveURL(/\/video-text-to-text$/);
    await expect(
      page.getByRole("heading", { name: "Video Text to Text" }),
    ).toBeVisible();
  });
});
