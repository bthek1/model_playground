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
    const { POSE_MODELS } = await import("../../src/vision/pose/types");
    const { MATTE_MODELS } = await import(
      "../../src/vision/backgroundRemoval"
    );
    const { SUPER_RES_MODELS } = await import("../../src/vision/superRes");
    const { VLM_MODELS, VIDEO_VLM_MODELS } = await import(
      "../../src/multimodal/types"
    );

    const ids = [
      ...ASR_MODELS,
      ...CLASSIFIER_MODELS,
      ...TTS_MODELS,
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
      ...MATTE_MODELS,
      ...SUPER_RES_MODELS,
      ...VLM_MODELS,
      ...VIDEO_VLM_MODELS,
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

  test("every bundled sample and gallery picture still resolves", async ({
    request,
  }) => {
    // Same class of bug as a dead model id, one layer up: these URLs are the
    // only reason a vision route works before the user has a file of their own,
    // and the unit suite mocks `fromUrl` away entirely. A 404 here is a page
    // whose sample buttons all fail — green tests, broken page.
    const { IMAGE_SAMPLES, TEXT_SAMPLES, PORTRAIT_SAMPLES } = await import(
      "../../src/vision/samples"
    );
    const { GALLERY_IMAGES } = await import("../../src/vision/gallery");

    const urls = [
      ...IMAGE_SAMPLES.map((s) => s.url),
      ...TEXT_SAMPLES.map((s) => s.url),
      ...PORTRAIT_SAMPLES.map((s) => s.url),
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

  test("every multimodal catalogue entry publishes the 4-bit graphs it asks for", async ({
    request,
  }) => {
    // The VLM catalogue does NOT take `loadOpts()`'s fp16/q8 — it takes
    // `vlmLoadOpts()`, which is q4f16 on WebGPU and q4 on WASM. Folding these
    // entries into the vision check above would ask the Hub for the wrong files
    // and pass or fail for the wrong reason, so they get their own defaults.
    //
    // Each entry loads three graphs, and all three must exist at the requested
    // precision: a repo that publishes `decoder_model_merged_q4f16.onnx` but not
    // `vision_encoder_q4f16.onnx` resolves fine on the API and then 404s
    // halfway through a 189 MB load.
    const { VLM_MODELS, VIDEO_VLM_MODELS } = await import(
      "../../src/multimodal/types"
    );

    const SUFFIX: Record<string, string> = { q4f16: "_q4f16", q4: "_q4" };
    const DEFAULT_DTYPE: Record<string, string> = {
      webgpu: "q4f16",
      wasm: "q4",
    };

    const missing: string[] = [];
    for (const model of [...VLM_MODELS, ...VIDEO_VLM_MODELS]) {
      const res = await request.get(
        `https://huggingface.co/api/models/${model.id}`,
      );
      const files: string[] = ((await res.json()).siblings ?? []).map(
        (f: { rfilename: string }) => f.rfilename,
      );
      const backends = model.backends ?? (["webgpu", "wasm"] as const);
      for (const backend of backends) {
        const dtype = model.dtypes?.[backend] ?? DEFAULT_DTYPE[backend];
        const suffix = SUFFIX[String(dtype)];
        if (suffix === undefined) continue;
        for (const graph of model.graphs) {
          const file = `onnx/${graph}${suffix}.onnx`;
          if (!files.includes(file)) {
            missing.push(`${model.id} (${backend}) -> ${file}`);
          }
        }
      }
    }
    expect(missing, "VLM graphs missing at the requested precision").toEqual([]);
  });

  test("the VLM catalogue quotes its real download size", async ({ request }) => {
    // These entries carry *measured* bytes rather than a params estimate,
    // because at q4f16 the precision is mixed: SmolVLM-256M's embedding table is
    // not 4-bit quantized at all, so an estimate is out by 30%. A measurement
    // that has drifted from the Hub is worse than an estimate, because the page
    // presents it as fact — so check it.
    const { VLM_MODELS, VIDEO_VLM_MODELS } = await import(
      "../../src/multimodal/types"
    );

    const wrong: string[] = [];
    for (const model of [...VLM_MODELS, ...VIDEO_VLM_MODELS]) {
      const res = await request.get(
        `https://huggingface.co/api/models/${model.id}/tree/main/onnx`,
      );
      const files: Array<{ path: string; size?: number; lfs?: { size?: number } }> =
        await res.json();
      const total = model.graphs.reduce((sum, graph) => {
        const entry = files.find(
          (f) => f.path === `onnx/${graph}_q4f16.onnx`,
        );
        return sum + (entry?.lfs?.size ?? entry?.size ?? 0);
      }, 0);
      const quoted = model.bytes.webgpu ?? 0;
      // 2% tolerance: the quoted figure is the sum of the graph files, and a
      // re-export that changes it by more than that is a real change the
      // catalogue should record.
      if (Math.abs(total - quoted) / quoted > 0.02) {
        wrong.push(`${model.id}: quoted ${quoted}, Hub says ${total}`);
      }
    }
    expect(wrong, "VLM sizes that have drifted from the Hub").toEqual([]);
  });
});
