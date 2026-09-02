import { AudioPage } from "../pages/AudioPage";
import { expect, test } from "../fixtures/base";

// Fast specs for the three in-browser audio routes. Every Hugging Face request
// is blocked, so these assert the route shell, the size-before-load guardrail,
// and the failure path — no weights, no ONNX Runtime, seconds not minutes.
//
// The specs that actually load a model live in `audio-models.spec.ts` (@slow).

/** Each route, its heading, and the models whose size warning should show. */
const ROUTES = [
  {
    path: "/asr",
    heading: "Automatic Speech Recognition",
    // Whisper-base is ~221 MB on the WASM path (fp32 decoder), Moonshine ~82 MB.
    large: "Whisper base",
    small: "Moonshine tiny",
  },
  {
    path: "/audio-classification",
    heading: "Audio Classification",
    // CLAP is ~292 MB at fp16; AST ~166 MB.
    large: "CLAP (zero-shot)",
    small: "AST (AudioSet)",
  },
  {
    path: "/text-to-speech",
    heading: "Text to Speech",
    // SpeechT5 is 144M params ⇒ ~275 MB at fp16; Kokoro is 82M ⇒ ~156 MB.
    large: "SpeechT5",
    small: "Kokoro 82M",
  },
] as const;

test.describe("audio routes", () => {
  // /text-to-audio is deliberately gated: MusicGen is 571 MB, so the route must
  // not fetch anything until the user opts in. That gate is the spec.
  test("/text-to-audio downloads nothing until the user opts in", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/text-to-audio");
    await expect(
      page.getByRole("heading", { name: "Text to Audio" }),
    ).toBeVisible();
    // The cost is stated before anything is fetched.
    await expect(page.getByText(/Experimental — and slow/)).toBeVisible();
    await expect(page.getByText(/autoregressive/)).toBeVisible();
    expect(hubRequests, "no weights before opt-in").toEqual([]);

    // Opting in is what starts the load.
    await audio.modelButton(/Download the model and continue/).click();
    await expect(audio.sizeNote).toBeVisible();
    await expect(page.getByLabel(/Prompt/)).toBeVisible();
  });

  // /audio-to-audio is the one route with no Transformers.js path — the DSP is
  // ours and the model is a bare ONNX graph. It is small (~8 MB), but it still
  // downloads on an explicit action, so the gate is the spec here too.
  test("/audio-to-audio downloads nothing until Load model is clicked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/audio-to-audio");
    await expect(
      page.getByRole("heading", { name: "Audio to Audio" }),
    ).toBeVisible();
    // The native rate is called out — every other audio route is 16 kHz.
    await expect(page.getByText(/48\s*kHz/).first()).toBeVisible();
    await expect(audio.sizeNote).toBeVisible();
    // 8 MB is nowhere near the guardrail, so it must stay quiet.
    await expect(audio.largeModelWarning).toHaveCount(0);
    expect(hubRequests, "no weights before the user asks").toEqual([]);

    // Transport is dead until a model is loaded.
    await expect(audio.modelButton(/Upload audio/)).toBeDisabled();
    await expect(audio.modelButton(/Record/)).toBeDisabled();

    await page.getByTestId("load-model").click();
    // With the Hub blocked the load must fail visibly rather than spin forever.
    await expect(audio.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length).toBeGreaterThan(0);
  });

  for (const route of ROUTES) {
    test(`${route.path} renders its shell without a model download`, async ({
      page,
      mockApi,
    }) => {
      await mockApi();
      const audio = new AudioPage(page);
      await audio.blockModelDownloads();
      await page.goto(route.path);

      await expect(
        page.getByRole("heading", { name: route.heading }),
      ).toBeVisible();
      // Both model choices are offered before anything is downloaded.
      await expect(audio.modelButton(route.large)).toBeVisible();
      await expect(audio.modelButton(route.small)).toBeVisible();
      // And the user is told the cost up front.
      await expect(audio.sizeNote).toBeVisible();
      await expect(audio.sizeNote).toContainText(/on WebGPU · .* on WASM/);
    });

    test(`${route.path} warns before a large download and not a small one`, async ({
      page,
      mockApi,
    }) => {
      await mockApi();
      const audio = new AudioPage(page);
      await audio.blockModelDownloads();
      await page.goto(route.path);

      // The guardrail must fire on the heavy model...
      await audio.modelButton(route.large).click();
      await expect(audio.largeModelWarning).toBeVisible();

      // ...and stay quiet on the light one, or it is just noise.
      await audio.modelButton(route.small).click();
      await expect(audio.largeModelWarning).toHaveCount(0);
    });

    test(`${route.path} surfaces a load failure instead of hanging`, async ({
      page,
      mockApi,
    }) => {
      await mockApi();
      const audio = new AudioPage(page);
      await audio.blockModelDownloads();
      await page.goto(route.path);

      // With the Hub unreachable the load must fail *visibly*: the page still
      // renders and says something, rather than spinning on "Loading model".
      await expect(audio.error).toBeVisible({ timeout: 30_000 });
      await expect(audio.readyStatus).toHaveCount(0);
      await expect(
        page.getByRole("heading", { name: route.heading }),
      ).toBeVisible();
    });
  }

  test("/asr keeps its transport controls disabled until a model is ready", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await audio.blockModelDownloads();
    await page.goto("/asr");

    // Nothing may be dispatched to a worker that has no model loaded.
    await expect(audio.modelButton(/Start listening/)).toBeDisabled();
    await expect(audio.modelButton(/Upload audio/)).toBeDisabled();
    await expect(audio.modelButton(/JFK/)).toBeDisabled();
  });
});
