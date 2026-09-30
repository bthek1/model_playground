import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FAMILIES, familyInfo, REGRESSION_FAMILIES } from "@/tabular/families";
import { MAX_BOOST_DEPTH, MAX_TREE_DEPTH } from "@/tabular/limits";
import type { Family, Hyperparams } from "@/tabular/types";

import { FamilyPicker } from "./FamilyPicker";

function setup(
  over: {
    families?: typeof FAMILIES;
    value?: Family;
    hp?: Hyperparams;
    rows?: number;
    classes?: number;
    disabled?: boolean;
  } = {},
) {
  const value = over.value ?? "boosting";
  const onChange = vi.fn();
  const onHp = vi.fn();
  const onSeed = vi.fn();
  render(
    <FamilyPicker
      families={over.families ?? FAMILIES}
      value={value}
      onChange={onChange}
      hp={over.hp ?? familyInfo(value).defaults}
      onHp={onHp}
      seed={42}
      onSeed={onSeed}
      rows={over.rows ?? 10_000}
      classes={over.classes ?? 2}
      disabled={over.disabled ?? false}
    />,
  );
  return { onChange, onHp, onSeed };
}

describe("FamilyPicker", () => {
  it("offers every rung of the list it is handed, with the current one pressed", () => {
    setup({ value: "forest" });
    for (const f of FAMILIES) {
      expect(screen.getByRole("button", { name: f.label })).toBeInTheDocument();
    }
    expect(screen.getByRole("button", { name: /random forest/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("renders the regression ladder when handed it, and not the classification one", () => {
    setup({ families: REGRESSION_FAMILIES, value: "ridge" });
    expect(screen.getByRole("button", { name: /ridge regression/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /quantile regression/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /logistic regression/i })).toBeNull();
  });

  it("reports a family change without touching the hyperparameters itself", () => {
    // Swapping the defaults is the *route's* job, because the defaults are
    // per-objective — this component must not guess.
    const { onChange, onHp } = setup();
    fireEvent.click(screen.getByRole("button", { name: /random forest/i }));
    expect(onChange).toHaveBeenCalledWith("forest");
    expect(onHp).not.toHaveBeenCalled();
  });

  it("says where the arithmetic runs, and why, for each rung", () => {
    // The one sentence this category exists to say out loud.
    setup({ value: "forest" });
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/runs on your cpu/i);
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/does not vectorise/i);
  });

  it("says GPU for the linear rungs", () => {
    setup({ value: "logistic" });
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/runs on your gpu/i);
  });

  it("quotes a fit estimate in words, on the training row count it was given", () => {
    setup({ value: "forest", rows: 10_000 });
    const note = screen.getByTestId("fit-estimate");
    expect(note).toHaveTextContent(/about 1 s/);
    expect(note).toHaveTextContent("10,000 training rows");
  });

  it("charges boosting per class, so a ten-class problem is quoted higher", () => {
    setup({ value: "boosting", rows: 10_000, classes: 2 });
    const binary = screen.getByTestId("fit-estimate").textContent ?? "";
    cleanup();
    setup({ value: "boosting", rows: 10_000, classes: 10 });
    const ten = screen.getByTestId("fit-estimate").textContent ?? "";
    expect(ten).not.toBe(binary);
    expect(ten).toMatch(/about (8|9) s/);
  });

  it("quotes nothing before there is a dataset to quote against", () => {
    setup({ rows: 0 });
    expect(screen.queryByTestId("fit-estimate")).toBeNull();
  });

  it("caps boosting's depth at the number Phase 0 measured, and the forest's higher", () => {
    setup({ value: "boosting" });
    expect(screen.getByLabelText(/max depth/i)).toHaveAttribute("max", String(MAX_BOOST_DEPTH));
    cleanup();
    setup({ value: "forest" });
    expect(screen.getByLabelText(/max depth/i)).toHaveAttribute("max", String(MAX_TREE_DEPTH));
  });

  it("clamps a depth carried over from a higher cap rather than showing it out of range", () => {
    // Switching forest → boosting with depth 10 held would otherwise render a
    // slider past its own max, which reads as the page losing the setting.
    setup({ value: "boosting", hp: { ...familyInfo("boosting").defaults, maxDepth: 10 } });
    expect(screen.getByLabelText(/max depth/i)).toHaveValue(String(MAX_BOOST_DEPTH));
  });

  it("shows only the knobs the chosen family reads", () => {
    setup({ value: "logistic" });
    expect(screen.getByLabelText(/epochs/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/^trees/i)).toBeNull();
  });

  it("reports a knob change as a whole hyperparameter object", () => {
    const { onHp } = setup({ value: "forest" });
    fireEvent.change(screen.getByLabelText(/^trees/i), { target: { value: "120" } });
    expect(onHp).toHaveBeenCalledWith(expect.objectContaining({ nTrees: 120 }));
  });

  it("puts the seed on screen and explains why it is there", () => {
    // A head-to-head between two families on two different splits compares the
    // splits, so the seed has to be visible and repeatable.
    const { onSeed } = setup();
    expect(screen.getByLabelText(/random seed/i)).toHaveValue(42);
    fireEvent.change(screen.getByLabelText(/random seed/i), { target: { value: "7" } });
    expect(onSeed).toHaveBeenCalledWith(7);
    expect(screen.getByText(/compares the splits/i)).toBeInTheDocument();
  });

  it("says nothing is fitted until Fit is pressed", () => {
    setup();
    expect(screen.getByText(/nothing is fitted until you press fit/i)).toBeInTheDocument();
  });

  it("disables every control while a fit is in flight", () => {
    setup({ disabled: true });
    expect(screen.getByRole("button", { name: /random forest/i })).toBeDisabled();
    expect(screen.getByLabelText(/max depth/i)).toBeDisabled();
    expect(screen.getByLabelText(/random seed/i)).toBeDisabled();
  });
});
