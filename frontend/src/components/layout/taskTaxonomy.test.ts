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

    expect(textGen.to).toBe("/playground");
    expect(textToSpeech.to).toBe("/tasks/text-to-speech");
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
  });

  it("returns undefined for an unknown path", () => {
    expect(categoryForPath("/nope")).toBeUndefined();
  });
});
