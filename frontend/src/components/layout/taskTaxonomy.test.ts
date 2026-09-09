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
    const textToAudio = all.find((t) => t.slug === "text-to-audio")!;
    const audioToAudio = all.find((t) => t.slug === "audio-to-audio")!;
    const vad = all.find((t) => t.slug === "voice-activity-detection")!;

    expect(textGen.to).toBe("/playground");
    expect(textToSpeech.to).toBe("/text-to-speech");
    expect(textToAudio.to).toBe("/text-to-audio");
    expect(audioToAudio.to).toBe("/audio-to-audio");
    expect(vad.to).toBe("/vad");

    // Unmapped tasks still fall through to the generic placeholder route.
    const discreteMaths = all.find((t) => t.slug === "discrete-maths")!;
    expect(discreteMaths.to).toBe("/tasks/discrete-maths");
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
    const classification = vision.tasks.find(
      (t) => t.slug === "image-classification",
    )!;
    const detection = vision.tasks.find((t) => t.slug === "object-detection")!;

    expect(classification.to).toBe("/image-classification");
    // The rest of the category is still research — they keep the placeholder
    // until their own route ships (see the sub-issues of #2).
    expect(detection.to).toBe("/tasks/object-detection");
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
  });

  it("returns undefined for an unknown path", () => {
    expect(categoryForPath("/nope")).toBeUndefined();
  });
});
