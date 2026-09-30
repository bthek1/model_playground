import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ClassificationMetrics } from "@/tabular/types";

import { MetricBlock } from "./MetricBlock";

function metrics(over: Partial<ClassificationMetrics> = {}): ClassificationMetrics {
  return {
    accuracy: 0.8,
    precision: 0.75,
    recall: 0.7,
    f1: 0.72,
    confusion: { counts: Int32Array.from([4, 1, 1, 4]), labels: ["no", "yes"] },
    baselineAccuracy: 0.55,
    baselineLabel: "no",
    ...over,
  };
}

describe("MetricBlock", () => {
  it("renders the four scores as percentages", () => {
    render(<MetricBlock metrics={metrics()} />);
    expect(screen.getByTestId("metric-block")).toHaveTextContent("80.0%");
    expect(screen.getByText(/precision/i)).toBeInTheDocument();
    expect(screen.getByText(/recall/i)).toBeInTheDocument();
    expect(screen.getByText("F1")).toBeInTheDocument();
  });

  it("marks the three macro averages as macro", () => {
    // Macro and micro differ most on exactly the imbalanced data this page is
    // about, so an unlabelled "precision" is an ambiguous number.
    render(<MetricBlock metrics={metrics()} />);
    expect(screen.getAllByText(/\(macro\)/i)).toHaveLength(3);
  });

  it("always states the baseline, beside the score and not under a toggle", () => {
    // /graph-classification's lesson: a class prior is the easiest thing in any
    // dataset to learn, so an accuracy without its null model is not a number.
    render(<MetricBlock metrics={metrics()} />);
    const note = screen.getByTestId("baseline-note");
    expect(note).toHaveTextContent("55.0%");
    expect(note).toHaveTextContent(/always answering/i);
    expect(note).toHaveTextContent("no");
  });

  it("states the margin in points rather than leaving the reader to subtract", () => {
    render(<MetricBlock metrics={metrics({ accuracy: 0.731, baselineAccuracy: 0.556 })} />);
    expect(screen.getByTestId("baseline-note")).toHaveTextContent("17.5 points above");
  });

  it("says plainly when the model is worse than guessing the majority", () => {
    render(<MetricBlock metrics={metrics({ accuracy: 0.4, baselineAccuracy: 0.55 })} />);
    const note = screen.getByTestId("baseline-note");
    expect(note).toHaveTextContent(/worse than guessing the majority/i);
    expect(note).toHaveTextContent("15.0 points below");
  });

  it("says the model learned the prior and nothing else when there is no margin", () => {
    // The case the whole component exists for: 59.8% on PROTEINS looks like a
    // result and is proof the model ignored its input.
    render(<MetricBlock metrics={metrics({ accuracy: 0.598, baselineAccuracy: 0.598 })} />);
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(
      /learned the class prior and nothing else/i,
    );
  });

  it("treats a sub-point difference as no margin rather than as a win", () => {
    // 0.5 points on a held-out half of a few hundred rows is noise, and calling
    // it "above" would dress noise as a finding.
    render(<MetricBlock metrics={metrics({ accuracy: 0.553, baselineAccuracy: 0.55 })} />);
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(/nothing else/i);
    expect(screen.getByTestId("baseline-note")).not.toHaveTextContent(/points above/);
  });
});
