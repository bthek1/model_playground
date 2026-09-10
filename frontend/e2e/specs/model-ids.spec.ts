import { expect, test } from "../fixtures/base";

// @slow, but the cheap half of it: a handful of Hub API calls, no weights.
//
// It lives in its own file, across every modality, because the bug it guards
// against is not modality-specific — two audio classification entries once
// pointed at `onnx-community/*` repos that do not exist, the Hub answered 401,
// and the unit suite stayed green because it mocks the network away. A model id
// is a string until something actually asks the Hub about it.
//
// Run it with `just fe-e2e-models` (seconds). Every new catalogue module gets
// added to the list below — that is the whole maintenance burden.

test.describe("@slow model catalogue", () => {
  test("every catalogue model id resolves on the Hugging Face Hub", async ({
    request,
  }) => {
    const { ASR_MODELS } = await import("../../src/audio/types");
    const { CLASSIFIER_MODELS } = await import("../../src/audio/classification");
    const { TTS_MODELS } = await import("../../src/audio/tts");
    const { MUSIC_MODELS } = await import("../../src/audio/textToAudio");
    const { ENHANCE_MODELS } = await import("../../src/audio/enhance/types");
    const { VAD_MODELS } = await import("../../src/audio/vad/types");
    const { IMAGE_CLASSIFIER_MODELS } = await import(
      "../../src/vision/classification"
    );
    const { DEPTH_MODELS } = await import("../../src/vision/depth");
    const { DETECTOR_MODELS } = await import("../../src/vision/detection");
    const { SEGMENTER_MODELS } = await import("../../src/vision/segmentation");
    const { ZERO_SHOT_MODELS } = await import("../../src/vision/zeroShot");
    const { ZERO_SHOT_DETECTOR_MODELS } = await import(
      "../../src/vision/zeroShotDetection"
    );
    const { FEATURE_MODELS } = await import("../../src/vision/features");
    const { SAM_MODELS } = await import("../../src/vision/sam/types");
    const { CAPTION_MODELS } = await import("../../src/vision/caption/types");
    const { POSE_MODELS } = await import("../../src/vision/pose/types");
    const { MATTE_MODELS } = await import(
      "../../src/vision/backgroundRemoval"
    );
    const { SUPER_RES_MODELS } = await import("../../src/vision/superRes");

    const ids = [
      ...ASR_MODELS,
      ...CLASSIFIER_MODELS,
      ...TTS_MODELS,
      ...MUSIC_MODELS,
      ...ENHANCE_MODELS,
      // The energy baseline has no repo to resolve — it is a detector, not a
      // checkpoint, so it is filtered out rather than asked about.
      ...VAD_MODELS.filter((m) => m.repo),
      ...IMAGE_CLASSIFIER_MODELS,
      ...DEPTH_MODELS,
      ...DETECTOR_MODELS,
      ...SEGMENTER_MODELS,
      ...ZERO_SHOT_MODELS,
      ...ZERO_SHOT_DETECTOR_MODELS,
      ...FEATURE_MODELS,
      ...SAM_MODELS,
      ...CAPTION_MODELS,
      ...MATTE_MODELS,
      ...SUPER_RES_MODELS,
      // A pose entry is a *pair*, so its own `id` is a composite that resolves
      // to nothing on the Hub — the two halves are what get downloaded.
      ...POSE_MODELS.flatMap((m) => [m.detector, m.pose]),
    ].map((m) => m.id);
    expect(ids.length).toBeGreaterThan(0);

    const bad: string[] = [];
    for (const id of ids) {
      const res = await request.get(`https://huggingface.co/api/models/${id}`);
      if (!res.ok()) bad.push(`${id} → ${res.status()}`);
    }
    expect(bad, "model ids that do not resolve on the Hub").toEqual([]);
  });

  test("every vision catalogue entry publishes the dtype its backend asks for", async ({
    request,
  }) => {
    // `loadOpts()` requests fp16 on WebGPU and q8 on WASM, and a catalogue entry
    // may override either (`VisionModel.dtypes`). A repo that publishes only
    // fp32 resolves fine on the API and then 404s at load time — which is why
    // `Xenova/mobilevitv2-1.0-imagenet1k-256` is not in the catalogue at all,
    // and why `mattmdjaga/segformer_b2_clothes` is pinned to fp32 on both
    // backends rather than left to the default. Assert the files, not merely
    // the repo.
    const { IMAGE_CLASSIFIER_MODELS } = await import(
      "../../src/vision/classification"
    );
    const { DEPTH_MODELS } = await import("../../src/vision/depth");
    const { DETECTOR_MODELS } = await import("../../src/vision/detection");
    const { SEGMENTER_MODELS } = await import("../../src/vision/segmentation");
    const { ZERO_SHOT_MODELS } = await import("../../src/vision/zeroShot");
    const { ZERO_SHOT_DETECTOR_MODELS } = await import(
      "../../src/vision/zeroShotDetection"
    );
    const { FEATURE_MODELS } = await import("../../src/vision/features");
    const { SAM_MODELS } = await import("../../src/vision/sam/types");
    const { CAPTION_MODELS } = await import("../../src/vision/caption/types");
    const { POSE_MODELS } = await import("../../src/vision/pose/types");
    const { MATTE_MODELS } = await import(
      "../../src/vision/backgroundRemoval"
    );
    const { SUPER_RES_MODELS } = await import("../../src/vision/superRes");

    // Suffix per precision, applied to each of the entry's graph base names.
    // Not every repo publishes one `model.onnx`: CLIP as a feature extractor
    // loads `vision_model.onnx` alone, and SAM ships two graphs. A check
    // hard-coded to `model.onnx` would look at the wrong file and pass, which
    // is precisely the silent failure this test exists to prevent —
    // `VisionModel.graphs` is what each entry declares instead.
    const SUFFIX: Record<string, string> = {
      fp16: "_fp16",
      q8: "_quantized",
      fp32: "",
    };
    const DEFAULT_DTYPE: Record<string, string> = { webgpu: "fp16", wasm: "q8" };

    const models = [
      ...IMAGE_CLASSIFIER_MODELS,
      ...DEPTH_MODELS,
      ...DETECTOR_MODELS,
      ...SEGMENTER_MODELS,
      ...ZERO_SHOT_MODELS,
      ...ZERO_SHOT_DETECTOR_MODELS,
      ...FEATURE_MODELS,
      ...SAM_MODELS,
      ...CAPTION_MODELS,
      ...MATTE_MODELS,
      ...SUPER_RES_MODELS,
      // Each half separately: the pair's `backends` gate applies to both, so it
      // is carried down here rather than declared twice in the catalogue.
      ...POSE_MODELS.flatMap((m) =>
        [m.detector, m.pose].map((stage) => ({ ...stage, backends: m.backends })),
      ),
    ];

    const missing: string[] = [];
    for (const model of models) {
      const res = await request.get(
        `https://huggingface.co/api/models/${model.id}`,
      );
      const files: string[] = ((await res.json()).siblings ?? []).map(
        (f: { rfilename: string }) => f.rfilename,
      );
      // Only the backends the entry claims: a WebGPU-only model is never asked
      // for on WASM, so a missing q8 export there is not a bug.
      const backends = model.backends ?? (["webgpu", "wasm"] as const);
      const graphs = model.graphs ?? (["model"] as const);
      for (const backend of backends) {
        const dtype = model.dtypes?.[backend] ?? DEFAULT_DTYPE[backend];
        const suffix = SUFFIX[String(dtype)];
        if (suffix === undefined) continue;
        for (const graph of graphs) {
          const file = `onnx/${graph}${suffix}.onnx`;
          if (!files.includes(file)) {
            missing.push(`${model.id} (${backend}) -> ${file}`);
          }
        }
      }
    }
    expect(missing, "catalogue entries missing a dtype we ask for").toEqual([]);
  });

  test("the BLIP mirror is still unusable, so the roadmap note stays honest", async ({
    request,
  }) => {
    // A *negative* assertion about a third-party repo, and deliberately so.
    // `docs/roadmaps/vision.md` §3.9 says BLIP is absent from `/image-to-text`
    // because its only ONNX mirror ships `split_0.onnx` / `split_1.onnx` rather
    // than the transformers.js layout (`encoder_model` + `decoder_model_merged`).
    // That claim ages: the day someone publishes a proper export this test fails,
    // which is exactly when we want to hear about it. A note in a doc cannot do
    // that.
    const res = await request.get(
      "https://huggingface.co/api/models/onnx-community/Salesforce_blip-image-captioning-base",
    );
    expect(res.ok(), "the BLIP mirror stopped resolving entirely").toBe(true);

    const files: string[] = ((await res.json()).siblings ?? []).map(
      (f: { rfilename: string }) => f.rfilename,
    );
    expect(
      files.some((f) => f.endsWith("encoder_model.onnx")),
      "BLIP now publishes a transformers.js layout — it can join the /image-to-text catalogue, and roadmap §3.9 needs updating",
    ).toBe(false);
    expect(files).toContain("split_0.onnx");
  });

  test("every bundled sample and gallery picture still resolves", async ({
    request,
  }) => {
    // Same class of bug as a dead model id, one layer up: these URLs are the
    // only reason a vision route works before the user has a file of their own,
    // and the unit suite mocks `fromUrl` away entirely. A 404 here is a page
    // whose sample buttons all fail — green tests, broken page.
    const { IMAGE_SAMPLES, TEXT_SAMPLES } = await import(
      "../../src/vision/samples"
    );
    const { GALLERY_IMAGES } = await import("../../src/vision/gallery");

    const urls = [
      ...IMAGE_SAMPLES.map((s) => s.url),
      ...TEXT_SAMPLES.map((s) => s.url),
      ...GALLERY_IMAGES.map((g) => g.url),
    ];
    expect(urls.length).toBeGreaterThan(0);

    const bad: string[] = [];
    for (const url of Array.from(new Set(urls))) {
      const res = await request.get(url);
      if (!res.ok()) bad.push(`${url} -> ${res.status()}`);
    }
    expect(bad, "sample images that do not resolve").toEqual([]);
  });
});
