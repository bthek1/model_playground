import { describe, expect, it } from "vitest";

import {
  categoryForPath,
  taskCategories,
  tasksBySlug,
} from "./taskTaxonomy";

describe("taskCategories", () => {
  it("exposes the eight top-level categories in order", () => {
    expect(taskCategories.map((c) => c.label)).toEqual([
      "Audio",
      "Computer Vision",
      "Multimodal",
      "Natural Language Processing",
      "Other",
      "Reinforcement Learning",
      "Tabular",
      "Theory",
    ]);
  });

  it("gives every category at least one task and an icon", () => {
    for (const category of taskCategories) {
      expect(category.tasks.length).toBeGreaterThan(0);
      expect(category.icon).toBeDefined();
    }
  });

  it("derives kebab-case slugs from task labels", () => {
    const audio = taskCategories.find((c) => c.label === "Audio")!;
    const asr = audio.tasks.find(
      (t) => t.label === "Automatic Speech Recognition",
    )!;
    expect(asr.slug).toBe("automatic-speech-recognition");
  });

  it("gives every task a globally unique slug", () => {
    const slugs = taskCategories.flatMap((c) => c.tasks.map((t) => t.slug));
    expect(new Set(slugs).size).toBe(slugs.length);
  });

  it("routes mapped tasks to their real route and the rest to /tasks/$slug", () => {
    const all = taskCategories.flatMap((c) => c.tasks);
    const textGen = all.find((t) => t.slug === "text-generation")!;
    const textToSpeech = all.find((t) => t.slug === "text-to-speech")!;
    const audioToAudio = all.find((t) => t.slug === "audio-to-audio")!;
    const vad = all.find((t) => t.slug === "voice-activity-detection")!;

    expect(textGen.to).toBe("/playground");
    expect(textToSpeech.to).toBe("/text-to-speech");
    expect(audioToAudio.to).toBe("/audio-to-audio");
    expect(vad.to).toBe("/vad");

    // Unmapped tasks still fall through to the generic placeholder route.
    const discreteMaths = all.find((t) => t.slug === "discrete-maths")!;
    expect(discreteMaths.to).toBe("/tasks/discrete-maths");
  });

  it("leaves the three heavy-only tasks on the placeholder", () => {
    // These had routes and lost them: every checkpoint each task has is a
    // several-hundred-megabyte download with no lighter alternative, which is
    // `adding-a-task-page.md` §0's second question answered "no". The rows stay
    // in the sidebar — the taxonomy mirrors the Hub, not our build state — but
    // they must not map to a route again without a smaller model to point at.
    const all = taskCategories.flatMap((c) => c.tasks);
    const at = (slug: string) => all.find((t) => t.slug === slug)!.to;

    expect(at("text-to-audio")).toBe("/tasks/text-to-audio");
    expect(at("image-to-text")).toBe("/tasks/image-to-text");
    expect(at("document-question-answering")).toBe(
      "/tasks/document-question-answering",
    );
  });

  it("maps the real Theory tools to their implemented routes", () => {
    const theory = taskCategories.find((c) => c.label === "Theory")!;
    const training = theory.tasks.find(
      (t) => t.slug === "linear-model-training",
    )!;
    const tensor = theory.tasks.find((t) => t.slug === "tensor-arithmetic")!;

    expect(training.label).toBe("Linear Model Training");
    expect(training.to).toBe("/training");
    expect(tensor.label).toBe("Tensor Arithmetic");
    expect(tensor.to).toBe("/tensor");
  });

  it("maps the implemented Computer Vision tasks to their real routes", () => {
    const vision = taskCategories.find((c) => c.label === "Computer Vision")!;
    const at = (slug: string) => vision.tasks.find((t) => t.slug === slug)!.to;

    expect(at("image-classification")).toBe("/image-classification");
    expect(at("depth-estimation")).toBe("/depth");
    expect(at("object-detection")).toBe("/object-detection");
    expect(at("image-segmentation")).toBe("/segmentation");
    expect(at("zero-shot-image-classification")).toBe(
      "/zero-shot-image-classification",
    );
    expect(at("zero-shot-object-detection")).toBe(
      "/zero-shot-object-detection",
    );
    // The one Computer Vision route whose path is not its slug — "image
    // features" is what the page is called, and `/image-feature-extraction`
    // reads like a spec section.
    expect(at("image-feature-extraction")).toBe("/image-features");
    expect(at("mask-generation")).toBe("/mask-generation");
    expect(at("keypoint-detection")).toBe("/pose");
    expect(at("video-classification")).toBe("/video-classification");
    // The one row in this category with no Hub task behind it: the Hub has no
    // `background-removal` tag, Transformers.js invented the pipeline, and #24
    // took the call to list it anyway rather than ship the route unlisted.
    expect(at("background-removal")).toBe("/background-removal");
    // Image to Image maps to the one part of it that runs in a tab: a single
    // forward pass of super-resolution. The diffusion half stays server-side.
    expect(at("image-to-image")).toBe("/super-resolution");
    // Image to 3D likewise covers the depth-to-cloud subset; full
    // reconstruction is diffusion and stays server-side.
    expect(at("image-to-3d")).toBe("/image-to-3d");

    // The rest of the category is still research — they keep the placeholder
    // until their own route ships (see the sub-issues of #2).
    expect(at("text-to-image")).toBe("/tasks/text-to-image");
    expect(at("image-to-video")).toBe("/tasks/image-to-video");
  });

  it("maps the implemented Audio tasks to their real routes", () => {
    const audio = taskCategories.find((c) => c.label === "Audio")!;
    const asr = audio.tasks.find(
      (t) => t.slug === "automatic-speech-recognition",
    )!;
    const classification = audio.tasks.find(
      (t) => t.slug === "audio-classification",
    )!;
    expect(asr.to).toBe("/asr");
    expect(classification.to).toBe("/audio-classification");
  });
});

describe("tasksBySlug", () => {
  it("indexes every task by its slug", () => {
    const count = taskCategories.reduce((n, c) => n + c.tasks.length, 0);
    expect(Object.keys(tasksBySlug)).toHaveLength(count);
    expect(tasksBySlug["object-detection"].label).toBe("Object Detection");
  });
});

describe("categoryForPath", () => {
  it("returns the category owning a mapped route", () => {
    expect(categoryForPath("/playground")).toBe(
      "Natural Language Processing",
    );
    expect(categoryForPath("/tensor")).toBe("Theory");
    expect(categoryForPath("/training")).toBe("Theory");
    expect(categoryForPath("/audio-classification")).toBe("Audio");
    expect(categoryForPath("/asr")).toBe("Audio");
    expect(categoryForPath("/image-classification")).toBe("Computer Vision");
    expect(categoryForPath("/depth")).toBe("Computer Vision");
    expect(categoryForPath("/object-detection")).toBe("Computer Vision");
    expect(categoryForPath("/segmentation")).toBe("Computer Vision");
  });

  it("returns undefined for an unknown path", () => {
    expect(categoryForPath("/nope")).toBeUndefined();
  });
});

describe("the Multimodal category", () => {
  const at = (slug: string) =>
    taskCategories
      .flatMap((c) => c.tasks)
      .find((t) => t.slug === slug)!.to;

  it("maps the three shipped routes", () => {
    // All three ride one engine — one worker, one hook, one `VlmRun` envelope —
    // and are separate routes because the Hub has separate tags and a user
    // looking for VQA does not click "Image Text to Text".
    expect(at("image-text-to-text")).toBe("/image-text-to-text");
    expect(at("visual-question-answering")).toBe("/visual-question-answering");
    expect(at("video-text-to-text")).toBe("/video-text-to-text");
  });

  it("leaves the rest of the category on the placeholder", () => {
    // Cut for size or blocked on NLP, each with measured numbers in
    // docs/roadmaps/multimodal.md. The rows stay because the sidebar mirrors
    // the Hub, not our build state.
    expect(at("audio-text-to-text")).toBe("/tasks/audio-text-to-text");
    expect(at("visual-document-retrieval")).toBe(
      "/tasks/visual-document-retrieval",
    );
    expect(at("image-text-to-image")).toBe("/tasks/image-text-to-image");
    expect(at("any-to-any")).toBe("/tasks/any-to-any");
  });
});

describe("Natural Language Processing", () => {
  const all = taskCategories.flatMap((c) => c.tasks);
  const at = (slug: string) => all.find((t) => t.slug === slug)!.to;

  it("maps the shipped NLP routes", () => {
    expect(at("text-classification")).toBe("/text-classification");
  });

  it("leaves Table Question Answering on the placeholder", () => {
    // §3.11's decision, pinned so it cannot be re-mapped without a model to
    // point at. TAPAS and TAPEX have no ONNX export and their table-aware
    // position embeddings mean a generic encoder is not a substitute;
    // text-to-SQL needs a ~1 GB coder model *and* a database. The row stays
    // in the sidebar because the taxonomy mirrors the Hub, not our build
    // state.
    expect(at("table-question-answering")).toBe(
      "/tasks/table-question-answering",
    );
  });
});
