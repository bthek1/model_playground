import { describe, expect, it } from "vitest";

import { categoryForPath, taskCategories } from "./taskTaxonomy";

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

  it("routes every task to its real route", () => {
    const all = taskCategories.flatMap((c) => c.tasks);
    const textGen = all.find((t) => t.slug === "text-generation")!;
    const textToSpeech = all.find((t) => t.slug === "text-to-speech")!;
    const audioToAudio = all.find((t) => t.slug === "audio-to-audio")!;
    const vad = all.find((t) => t.slug === "voice-activity-detection")!;

    // Was `/playground` until §3.8 shipped a real task page for it. The demo
    // surface stays reachable on its own terms; it no longer owns the row.
    expect(textGen.to).toBe("/text-generation");
    expect(textToSpeech.to).toBe("/text-to-speech");
    expect(audioToAudio.to).toBe("/audio-to-audio");
    expect(vad.to).toBe("/vad");

    // #59: there is no placeholder route any more, so nothing may point at one.
    for (const t of all) expect(t.to, t.slug).not.toMatch(/^\/tasks\//);
  });

  it("lists no Hub task that has no page", () => {
    // #59. These rows used to stay in the sidebar and fall through to a
    // "/tasks/$slug — on the roadmap" placeholder, for tasks the roadmaps had
    // already measured and ruled out: three built and cut for size, the rest
    // diffusion-sized or with no export. The reasons live in docs/roadmaps/;
    // the sidebar lists only what it can open. Re-adding one needs a route.
    const slugs = new Set(taskCategories.flatMap((c) => c.tasks.map((t) => t.slug)));
    for (const slug of [
      "text-to-audio",
      "text-to-image",
      "image-to-text",
      "image-to-video",
      "unconditional-image-generation",
      "text-to-video",
      "text-to-3d",
      "video-to-video",
      "audio-text-to-text",
      "image-text-to-image",
      "image-text-to-video",
      "document-question-answering",
      "visual-document-retrieval",
      "any-to-any",
      "table-question-answering",
    ]) {
      expect(slugs.has(slug), slug).toBe(false);
    }
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

  // #56: Theory's last row.
  it("maps Discrete Maths to its route", () => {
    const theory = taskCategories.find((c) => c.label === "Theory")!;
    expect(theory.tasks.find((t) => t.slug === "discrete-maths")!.to).toBe("/discrete-maths");
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

describe("categoryForPath", () => {
  it("returns the category owning a mapped route", () => {
    // `/playground` belongs to **Theory** now, not to NLP. §3.8 took the
    // Text Generation row off it — a WebGPU demo surface is not a task page —
    // and gave the demo a Theory row of its own so it stays in the sidebar
    // rather than becoming URL-only.
    expect(categoryForPath("/playground")).toBe("Theory");
    expect(categoryForPath("/text-generation")).toBe(
      "Natural Language Processing",
    );
    expect(categoryForPath("/tensor")).toBe("Theory");
    expect(categoryForPath("/training")).toBe("Theory");
    expect(categoryForPath("/discrete-maths")).toBe("Theory");
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
});

describe("Natural Language Processing", () => {
  const all = taskCategories.flatMap((c) => c.tasks);
  const at = (slug: string) => all.find((t) => t.slug === slug)!.to;

  it("maps the shipped NLP routes", () => {
    expect(at("text-classification")).toBe("/text-classification");
    expect(at("token-classification")).toBe("/token-classification");
    expect(at("question-answering")).toBe("/question-answering");
    expect(at("zero-shot-classification")).toBe("/zero-shot-classification");
    expect(at("fill-mask")).toBe("/fill-mask");
    // §3.7 ships as a pair: one engine and one catalogue behind two rows.
    // `feature-extraction` points at `/text-features`, mirroring
    // `/image-features` rather than taking the slug's own name.
    expect(at("feature-extraction")).toBe("/text-features");
    expect(at("sentence-similarity")).toBe("/sentence-similarity");
    // §3.5, the category's first seq2seq page: one pair at a time, because a
    // Marian checkpoint *is* its direction.
    expect(at("translation")).toBe("/translation");
    // §3.6, and the page that survived its own Phase 0 gate: distilbart opens a
    // q8 WebGPU session, which is the only configuration inside the size bar.
    expect(at("summarization")).toBe("/summarization");
    // §3.8, and the row **moved**: it used to resolve to `/playground`, which
    // is a WebGPU demo surface rather than a task page. A page that is not a
    // task page should not claim a task row.
    expect(at("text-generation")).toBe("/text-generation");
    expect(at("text-generation")).not.toBe("/playground");
    // §3.10, the most complete page in the category: all four retrieval stages
    // client-side, two of them with no model at all.
    expect(at("text-ranking")).toBe("/text-ranking");
  });
});

describe("the Tabular category", () => {
  const all = taskCategories.flatMap((c) => c.tasks);
  const at = (slug: string) => all.find((t) => t.slug === slug)!.to;

  it("maps all three of its rows, because the category shipped whole", () => {
    // The only category with no checkpoint anywhere in it: the model is fitted
    // in the tab on the user's own CSV, so the feasibility bar is the *fit*
    // rather than the download, and every row cleared it.
    expect(at("tabular-classification")).toBe("/tabular-classification");
    expect(at("tabular-regression")).toBe("/tabular-regression");
    expect(at("time-series-forecasting")).toBe("/time-series-forecasting");
  });

  it("lists exactly its three rows", () => {
    const tabular = taskCategories.find((c) => c.label === "Tabular")!;
    expect(tabular.tasks).toHaveLength(3);
  });

  it("puts every tabular route in the Tabular category, not in Theory", () => {
    // These pages train in the tab, which makes them look like the Theory
    // surfaces (`/training`, `/tensor`). They are not: they answer Hub task
    // rows and take the user's own data, so they belong to the taxonomy's own
    // category rather than to this repo's invented one.
    for (const path of [
      "/tabular-classification",
      "/tabular-regression",
      "/time-series-forecasting",
    ]) {
      expect(categoryForPath(path), path).toBe("Tabular");
    }
  });
});

describe("the Reinforcement Learning category", () => {
  const at = (slug: string) =>
    taskCategories.flatMap((c) => c.tasks).find((t) => t.slug === slug)!.to;

  it("maps both rows — the category is complete as scoped", () => {
    // `/rl` holds three algorithms on two environments; `/robotics` holds the
    // grounding pair and the behaviour-cloning entry (docs/roadmaps/rl.md).
    expect(at("reinforcement-learning")).toBe("/rl");
    expect(at("robotics")).toBe("/robotics");
  });

  it("files both routes under the category in the sidebar", () => {
    expect(categoryForPath("/rl")).toBe("Reinforcement Learning");
    expect(categoryForPath("/robotics")).toBe("Reinforcement Learning");
  });
});

