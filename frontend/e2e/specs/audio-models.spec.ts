import { AudioPage } from "../pages/AudioPage";
import { MIN_SDR_GAIN_DB, measureEnhancement } from "../utils/enhance";
import { expect, test } from "../fixtures/base";

// @slow — these download real ONNX weights from Hugging Face (80-220 MB each)
// and open a real ONNX Runtime session. Excluded from the default run; opt in
// with `just fe-e2e-slow` (E2E_SLOW=1).
//
// They exist because two bugs shipped past a fully green unit suite on
// 2026-09-01, and both lived precisely in what the unit tests mock away:
//
//   1. Every ASR model failed to open a WASM session — the quantized decoders
//      hit an ONNX Runtime QDQ bug. The *universal fallback* could not load a
//      model at all, and no mocked test could tell.
//   2. Two classification models pointed at Hugging Face repos that don't
//      exist (401). Only a real fetch reveals that.
//
// So the assertions here are deliberately end-to-end: the model must actually
// become ready, and for ASR it must produce the right words.
//
// Note the `audio.load()` before every `waitForReady`: since the four-slot
// migration nothing downloads on navigation, so a spec that wants weights has to
// press the LOAD slot's button like a user would.

const DOWNLOAD_BUDGET_MS = 10 * 60 * 1000;

test.describe("@slow real model loads", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/asr loads Whisper-base and transcribes the JFK sample", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/asr");

    // Regression guard for bug 1: on a machine with no GPU this is the WASM
    // path, which is exactly what was broken.
    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);
    expect(["webgpu", "wasm"]).toContain(await audio.backend());

    await audio.modelButton(/JFK/).click();
    // The reference transcript for the clip — a model that loads but decodes
    // garbage is still broken, so assert the words, not just "some text".
    await expect(
      page.getByText(/ask not what your country can do for you/i),
    ).toBeVisible({ timeout: DOWNLOAD_BUDGET_MS });
  });

  test("/asr renders the transcript as timestamped segments", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/asr");
    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);

    await audio.modelButton(/JFK/).click();

    // Scope to the Transcript card — the sample card also quotes the reference.
    const transcript = page
      .locator('[data-slot="card"]')
      .filter({ hasText: /^Transcript/ })
      .last();
    const segments = transcript.locator("ol li");

    // The plan claimed timestamps once before while the route rendered plain
    // text; this is the guard against that regressing again silently.
    await expect(segments.first()).toBeVisible({ timeout: DOWNLOAD_BUDGET_MS });
    await expect(segments.first()).toContainText(/^\d+:\d\d/);
    await expect(transcript).toContainText(/ask not what your country/i);
  });

  test("/asr shows the warm-up state between download and ready", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/asr");

    // Weights are cached from the previous test, so warm-up is the visible
    // phase here. Either state proves the load is progressing, then ready must
    // follow — the point is that warm-up never wedges the load.
    await expect(audio.warmingStatus.or(audio.loadingStatus).first()).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);
  });

  test("/audio-classification loads every model in the catalogue", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/audio-classification");

    // Regression guard for bug 2: a wrong repo id 401s and never reaches ready.
    // Switching models also exercises the one-model-live dispose path.
    for (const model of ["AST (AudioSet)", "wav2vec2 keyword spotting", "CLAP (zero-shot)"]) {
      await audio.modelButton(model).click();
      await audio.load();
      await audio.waitForReady(DOWNLOAD_BUDGET_MS);
      await expect(audio.error).toHaveCount(0);
    }
  });

  test("/text-to-speech loads Kokoro and synthesises audio", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/text-to-speech");
    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);

    await audio.modelButton(/^Speak/).click();
    // The result card only renders once real samples came back from the worker.
    await expect(page.getByText(/Generated audio/)).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    await expect(page.getByText(/\d+\.\d+s · \d+ kHz/)).toBeVisible();
  });
});

// The enhancement route is a different kind of risk from the others. Its model
// loads or doesn't like any other, but its *DSP* is ours — and wrong DSP does
// not throw, it produces plausible audio with artefacts. The unit tests pin the
// pipeline against arrays captured from the reference implementation; these
// close the loop by running the real graph in a real browser and measuring.
//
// This file covers the WASM path — the universal fallback, and the one the
// `["webgpu", "wasm"]` provider list has to reach on its own. The WebGPU half
// lives in `webgpu/enhance.spec.ts`, which needs a real GPU.
test.describe("@slow speech enhancement", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("enhances a noisy clip on wasm", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/audio-to-audio");

    const { before, after, samples } = await measureEnhancement(page, "wasm");

    expect(samples).toBeGreaterThan(0);
    expect(before).toBeLessThan(1);
    expect(after - before).toBeGreaterThan(MIN_SDR_GAIN_DB);
  });

  test("the route enhances an uploaded file end to end", async ({
    page,
    mockApi,
    request,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/audio-to-audio");

    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);
    expect(["webgpu", "wasm"]).toContain(await audio.backend());

    const clip = await request.get(
      "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: "jfk.wav",
      mimeType: "audio/wav",
      buffer: await clip.body(),
    });

    // Both rows only render once real samples came back from the worker.
    await expect(page.getByText("Noisy input")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    await expect(page.getByText("Enhanced")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    await expect(audio.error).toHaveCount(0);
  });
});

// VAD is the one audio task whose correctness a mocked test genuinely cannot
// reach. The frame loop feeds the model a 576-sample window (64 samples of
// context + a 512-sample frame); feeding a bare 512 runs perfectly happily and
// returns numbers that simply never cross any threshold. A unit test with a fake
// session cannot tell the two apart — only real weights can.
test.describe("@slow voice activity detection", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/vad finds the speech in the JFK clip", async ({
    page,
    mockApi,
    request,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/vad");

    await audio.load();
    await audio.waitForReady(DOWNLOAD_BUDGET_MS);
    // Silero runs on WASM by design — see src/audio/vad/session.ts. A page
    // reporting WebGPU here means the provider list was loosened.
    expect(await audio.backend()).toBe("wasm");

    const clip = await request.get(
      "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: "jfk.wav",
      mimeType: "audio/wav",
      buffer: await clip.body(),
    });

    // The clip is 11 s of continuous speech with a short lead-in, so a working
    // detector finds a few seconds of speech in a handful of segments. A frame
    // loop without the context window scores every frame near zero and this
    // summary reads "0 segments · 0.0s speech".
    const summary = page.getByText(/segments? · [\d.]+s speech/);
    await expect(summary).toBeVisible({ timeout: DOWNLOAD_BUDGET_MS });

    const text = (await summary.textContent()) ?? "";
    const [, segments, speech, total] =
      /(\d+) segments? · ([\d.]+)s speech of ([\d.]+)s/.exec(text) ?? [];
    expect(Number(segments)).toBeGreaterThan(0);
    expect(Number(speech)).toBeGreaterThan(Number(total) * 0.5);
    await expect(audio.error).toHaveCount(0);
  });

  test("the energy baseline runs with no download at all", async ({
    page,
    mockApi,
    request,
  }) => {
    await mockApi();
    const audio = new AudioPage(page);
    await page.goto("/vad");

    // Nothing from the Hub may be fetched for this detector — it has no weights.
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests++;
        return route.continue();
      },
    );

    await audio.modelButton(/Energy VAD/).click();
    await audio.load();
    await audio.waitForReady(30_000);

    const clip = await request.get(
      "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: "jfk.wav",
      mimeType: "audio/wav",
      buffer: await clip.body(),
    });

    await expect(page.getByText(/segments? · [\d.]+s speech/)).toBeVisible({
      timeout: 60_000,
    });
    expect(hubRequests).toBe(0);
  });
});

test.describe("@slow model catalogue", () => {
  test("every catalogue model id resolves on the Hugging Face Hub", async ({
    request,
  }) => {
    // The cheap half of the @slow group: five HEAD-ish API calls, no weights.
    // This is the check that would have caught the 401 repos in seconds.
    const { ASR_MODELS } = await import("../../src/audio/types");
    const { CLASSIFIER_MODELS } = await import("../../src/audio/classification");
    const { TTS_MODELS } = await import("../../src/audio/tts");
    const { MUSIC_MODELS } = await import("../../src/audio/textToAudio");
    const { ENHANCE_MODELS } = await import("../../src/audio/enhance/types");
    const { VAD_MODELS } = await import("../../src/audio/vad/types");

    const ids = [
      ...ASR_MODELS,
      ...CLASSIFIER_MODELS,
      ...TTS_MODELS,
      ...MUSIC_MODELS,
      ...ENHANCE_MODELS,
      // The energy baseline has no repo to resolve — it is a detector, not a
      // checkpoint, so it is filtered out rather than asked about.
      ...VAD_MODELS.filter((m) => m.repo),
    ].map((m) => m.id);
    expect(ids.length).toBeGreaterThan(0);

    const bad: string[] = [];
    for (const id of ids) {
      const res = await request.get(`https://huggingface.co/api/models/${id}`);
      if (!res.ok()) bad.push(`${id} → ${res.status()}`);
    }
    expect(bad, "model ids that do not resolve on the Hub").toEqual([]);
  });
});
