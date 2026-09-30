import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { parseCsv } from "@/tabular/csv";
import type { Objective } from "@/tabular/types";

import { ColumnPicker } from "./ColumnPicker";

const dataset = parseCsv(
  "age,city,score,defaulted\n30,north,1.5,yes\n40,south,2.5,no\n50,north,,yes\n",
).dataset;

function setup(objective: Objective, target = 3, features = [0, 1, 2]) {
  const onTarget = vi.fn();
  const onFeatures = vi.fn();
  render(
    <ColumnPicker
      dataset={dataset}
      objective={objective}
      target={target}
      onTarget={onTarget}
      features={features}
      onFeatures={onFeatures}
    />,
  );
  return { onTarget, onFeatures };
}

describe("ColumnPicker", () => {
  it("offers only categorical columns as a classification target", () => {
    // A numeric column with 300 distinct values is a regression problem wearing
    // the wrong hat, and would produce 300 classes of one row each.
    setup("classification");
    const options = within(screen.getByLabelText(/target/i)).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["city (2 values)", "defaulted (2 values)"]);
  });

  it("offers only numeric columns as a regression target", () => {
    setup("regression", 2, [0, 1, 3]);
    const options = within(screen.getByLabelText(/target/i)).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["age", "score"]);
  });

  it("never offers the target as a feature", () => {
    // A model handed its own answer scores 1.00 and teaches nothing — the one
    // rule here that fails by looking like a spectacular success.
    setup("classification");
    const picker = screen.getByTestId("column-picker");
    expect(within(picker).queryByRole("button", { name: /^defaulted/ })).toBeNull();
    expect(within(picker).getByRole("button", { name: /^age/ })).toBeInTheDocument();
  });

  it("reports a change of target", () => {
    const { onTarget } = setup("classification");
    fireEvent.change(screen.getByLabelText(/target/i), { target: { value: "1" } });
    expect(onTarget).toHaveBeenCalledWith(1);
  });

  it("toggles a feature off and back on, keeping the list sorted", () => {
    const { onFeatures } = setup("classification");
    fireEvent.click(screen.getByRole("button", { name: /^city/ }));
    expect(onFeatures).toHaveBeenLastCalledWith([0, 2]);

    onFeatures.mockClear();
    render(
      <ColumnPicker
        dataset={dataset}
        objective="classification"
        target={3}
        onTarget={vi.fn()}
        features={[2, 0]}
        onFeatures={onFeatures}
      />,
    );
    fireEvent.click(screen.getAllByRole("button", { name: /^city/ })[1]);
    expect(onFeatures).toHaveBeenLastCalledWith([0, 1, 2]);
  });

  it("selects all features without ever including the target", () => {
    const { onFeatures } = setup("classification", 1, []);
    fireEvent.click(screen.getByRole("button", { name: /^all$/i }));
    expect(onFeatures).toHaveBeenCalledWith([0, 2, 3]);
  });

  it("clears the feature set, so the page can say what is missing", () => {
    const { onFeatures } = setup("classification");
    fireEvent.click(screen.getByRole("button", { name: /^none$/i }));
    expect(onFeatures).toHaveBeenCalledWith([]);
  });

  it("marks which feature columns have gaps, before anything is fitted", () => {
    // The imputation and the is-missing indicator are both decisions the user
    // should see coming rather than discover in a coefficient list.
    setup("classification");
    expect(screen.getByRole("button", { name: /score.*1 missing/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^age$/ })).toBeInTheDocument();
  });

  it("counts the chosen features against the ones available", () => {
    setup("classification", 3, [0, 2]);
    expect(screen.getByText(/features — 2 of 3/i)).toBeInTheDocument();
  });

  it("says what to do when the file has no eligible target at all", () => {
    const numericOnly = parseCsv("a,b\n1,2\n3,4\n").dataset;
    render(
      <ColumnPicker
        dataset={numericOnly}
        objective="classification"
        target={-1}
        onTarget={vi.fn()}
        features={[]}
        onFeatures={vi.fn()}
      />,
    );
    expect(screen.getByText(/no text column in this file to classify/i)).toBeInTheDocument();
  });
});
