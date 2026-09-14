import { describe, expect, it } from "vitest";

import { titleForPath } from "./documentTitle";
import { taskCategories } from "@/components/layout/taskTaxonomy";

describe("titleForPath", () => {
  it("gives the landing page the bare product name", () => {
    expect(titleForPath("/")).toBe("Model Playground");
  });

  it("suffixes static routes", () => {
    expect(titleForPath("/home")).toBe("Home · Model Playground");
    expect(titleForPath("/login")).toBe("Sign in · Model Playground");
    expect(titleForPath("/signup")).toBe("Create account · Model Playground");
  });

  it("names a real task route from the taxonomy", () => {
    expect(titleForPath("/asr")).toBe(
      "Automatic Speech Recognition · Model Playground",
    );
    expect(titleForPath("/object-detection")).toBe(
      "Object Detection · Model Playground",
    );
  });

  it("names a route whose slug differs from its path", () => {
    // `image-to-image` -> /super-resolution: the title follows the taxonomy
    // label, not the URL.
    expect(titleForPath("/super-resolution")).toBe(
      "Image to Image · Model Playground",
    );
  });

  it("names the generic /tasks/$slug placeholder", () => {
    const placeholder = taskCategories
      .flatMap((c) => c.tasks)
      .find((t) => t.to.startsWith("/tasks/"));
    expect(placeholder, "expected at least one unimplemented task").toBeDefined();
    expect(titleForPath(placeholder!.to)).toBe(
      `${placeholder!.label} · Model Playground`,
    );
  });

  it("tolerates a trailing slash", () => {
    expect(titleForPath("/home/")).toBe("Home · Model Playground");
  });

  it("falls back to the product name rather than inventing a label", () => {
    expect(titleForPath("/nope")).toBe("Model Playground");
    expect(titleForPath("/tasks/not-a-task")).toBe("Model Playground");
  });

  it("gives every implemented task route a distinct title", () => {
    const real = taskCategories
      .flatMap((c) => c.tasks)
      .filter((t) => !t.to.startsWith("/tasks/"));
    const titles = real.map((t) => titleForPath(t.to));
    expect(new Set(titles).size).toBe(real.length);
    expect(titles.every((t) => t !== "Model Playground")).toBe(true);
  });
});
