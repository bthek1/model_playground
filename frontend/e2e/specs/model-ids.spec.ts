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

    const FILE_FOR: Record<string, string> = {
      fp16: "onnx/model_fp16.onnx",
      q8: "onnx/model_quantized.onnx",
      fp32: "onnx/model.onnx",
    };
    const DEFAULT_DTYPE = { webgpu: "fp16", wasm: "q8" } as const;

    const models = [
      ...IMAGE_CLASSIFIER_MODELS,
      ...DEPTH_MODELS,
      ...DETECTOR_MODELS,
      ...SEGMENTER_MODELS,
      ...ZERO_SHOT_MODELS,
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
      for (const backend of backends) {
        const dtype = model.dtypes?.[backend] ?? DEFAULT_DTYPE[backend];
        const file = FILE_FOR[String(dtype)];
        if (file && !files.includes(file)) {
          missing.push(`${model.id} (${backend}) -> ${file}`);
        }
      }
    }
    expect(missing, "catalogue entries missing a dtype we ask for").toEqual([]);
  });
});
