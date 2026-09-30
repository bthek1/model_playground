import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Fact, Score } from "./readout";

describe("readout cells", () => {
  it("renders each as a term and its value, for the caller's <dl>", () => {
    render(
      <dl>
        <Fact label="Nodes" value="2708" />
        <Score label="Test accuracy" value="81.2%" />
      </dl>,
    );
    expect(screen.getAllByRole("term").map((t) => t.textContent)).toEqual([
      "Nodes",
      "Test accuracy",
    ]);
    expect(screen.getAllByRole("definition").map((d) => d.textContent)).toEqual([
      "2708",
      "81.2%",
    ]);
  });

  it("sets numbers in tabular figures, so a changing value does not jitter", () => {
    render(
      <dl>
        <Fact label="Edges" value="10556" />
        <Score label="AUC" value="0.912" />
      </dl>,
    );
    for (const dd of screen.getAllByRole("definition")) {
      expect(dd).toHaveClass("tabular-nums");
    }
  });
});
