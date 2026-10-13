import { describe, expect, it } from "vitest";

import { titleForPath } from "./documentTitle";
import { taskCategories } from "@/components/layout/taskTaxonomy";

describe("titleForPath", () => {
  it("gives the landing page the bare product name", () => {
    expect(titleForPath("/")).toBe("Model Playground");
  });

  it("suffixes static routes", () => {
    expect(titleForPath("/privacy")).toBe("Privacy · Model Playground");
    expect(titleForPath("/login")).toBe("Sign in · Model Playground");
    expect(titleForPath("/signup")).toBe("Create account · Model Playground");
  });

  it("says what the home page is, not just its name", () => {
    expect(titleForPath("/home")).toBe(
      "Model Playground: ML models in your browser, on your GPU",
    );
  });

  // #71: a task's tab title is its search title — the words people search
  // with, which is why /asr is "Speech to Text" rather than the Hub's name.
  it("names a real task route by its search title", () => {
    expect(titleForPath("/asr")).toBe(
      "Speech to Text with Whisper, in your browser · Model Playground",
    );
    expect(titleForPath("/object-detection")).toBe(
      "Object Detection in your browser, on your GPU · Model Playground",
    );
  });

  it("names a route by what it does, not by its Hub slug", () => {
    // `image-to-image` -> /super-resolution: nobody searches "Image to Image"
    // for an upscaler.
    expect(titleForPath("/super-resolution")).toBe(
      "AI Image Upscaler (Swin2SR) in your browser · Model Playground",
    );
  });

  it("tolerates a trailing slash and upper case", () => {
    expect(titleForPath("/privacy/")).toBe("Privacy · Model Playground");
    expect(titleForPath("/Login")).toBe("Sign in · Model Playground");
  });

  it("falls back to the product name rather than inventing a label", () => {
    expect(titleForPath("/nope")).toBe("Model Playground");
    // The retired /tasks/$slug placeholder (#59) is just an unknown path now.
    expect(titleForPath("/tasks/text-to-image")).toBe("Model Playground");
  });

  it("gives every task route a distinct title", () => {
    const real = taskCategories.flatMap((c) => c.tasks);
    const titles = real.map((t) => titleForPath(t.to));
    expect(new Set(titles).size).toBe(real.length);
    expect(titles.every((t) => t !== "Model Playground")).toBe(true);
  });
});
