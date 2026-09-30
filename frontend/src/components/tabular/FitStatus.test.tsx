import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FitPartial, FitResult } from "@/tabular/types";

import { FitStatus } from "./FitStatus";

const result = {
  fitMs: 1234,
  trainRows: 1350,
  compute: "cpu",
} as FitResult;

function setup(
  over: {
    canFit?: boolean;
    running?: boolean;
    partial?: FitPartial | null;
    result?: FitResult | null;
    error?: string | null;
    blockedReason?: string | null;
  } = {},
) {
  const onFit = vi.fn();
  const onStop = vi.fn();
  render(
    <FitStatus
      canFit={over.canFit ?? true}
      running={over.running ?? false}
      partial={over.partial ?? null}
      result={over.result ?? null}
      error={over.error ?? null}
      onFit={onFit}
      onStop={onStop}
      blockedReason={over.blockedReason ?? null}
    />,
  );
  return { onFit, onStop };
}

describe("FitStatus", () => {
  it("offers Fit, and nothing else, before anything has run", () => {
    const { onFit } = setup();
    const button = screen.getByTestId("fit-button");
    expect(button).toHaveTextContent(/^Fit model$/);
    expect(button).toBeEnabled();
    expect(screen.queryByTestId("fit-progress")).toBeNull();
    expect(screen.queryByTestId("fit-stop")).toBeNull();
    expect(screen.queryByTestId("model-ready")).toBeNull();
    fireEvent.click(button);
    expect(onFit).toHaveBeenCalledOnce();
  });

  it("disables Fit with the reason in the user's terms", () => {
    // A greyed-out button with no reason attached is a dead end.
    setup({ canFit: false, blockedReason: "Choose a sample or drop a CSV to fit a model." });
    expect(screen.getByTestId("fit-button")).toBeDisabled();
    expect(screen.getByText(/choose a sample or drop a csv/i)).toBeInTheDocument();
  });

  it("renders a determinate bar, not a spinner, because the total is known", () => {
    // `model/progress.ts`'s indeterminate mode is the obvious reach for a page
    // with no bytes and is the wrong one: epochs, trees and rows are
    // hyperparameters the user set a moment ago.
    setup({
      running: true,
      partial: { done: 30, total: 120, phase: "Fitting", loss: 0.4213 },
    });
    const bar = screen.getByTestId("fit-progress");
    expect(bar).toHaveTextContent("Fitting · 30/120");
    expect(bar).toHaveTextContent("loss 0.4213");
    expect(bar.querySelector("[style*='width: 25%']")).not.toBeNull();
  });

  it("names the phase, so a long permutation pass is not read as a stall", () => {
    setup({
      running: true,
      partial: { done: 3, total: 9, phase: "Permuting columns", loss: null },
    });
    expect(screen.getByTestId("fit-progress")).toHaveTextContent("Permuting columns · 3/9");
    // No loss for a phase that has none — a "loss null" would be worse than absent.
    expect(screen.getByTestId("fit-progress")).not.toHaveTextContent(/loss/);
  });

  it("offers Stop only while a fit is in flight", () => {
    const { onStop } = setup({
      running: true,
      partial: { done: 1, total: 60, phase: "Fitting", loss: 1 },
    });
    fireEvent.click(screen.getByTestId("fit-stop"));
    expect(onStop).toHaveBeenCalledOnce();
    expect(screen.getByTestId("fit-button")).toBeDisabled();
    expect(screen.getByTestId("fit-button")).toHaveTextContent(/fitting/i);
  });

  it("emits model-ready when a fit finishes, with its duration and where it ran", () => {
    // The shared testid, asked of a page with nothing to download: "is there
    // something to run?" is the same question.
    setup({ result });
    const ready = screen.getByTestId("model-ready");
    expect(ready).toHaveTextContent("Fitted in 1.2s on 1,350 rows");
    expect(ready).toHaveTextContent("CPU");
  });

  it("says GPU when that is where the fit actually ran", () => {
    setup({ result: { ...result, compute: "gpu" } as FitResult });
    expect(screen.getByTestId("model-ready")).toHaveTextContent("GPU");
  });

  it("invites a refit once there is a result, rather than repeating Fit model", () => {
    setup({ result });
    expect(screen.getByTestId("fit-button")).toHaveTextContent(/fit again/i);
  });

  it("hides the finished line while the next fit is running", () => {
    // Otherwise the previous fit's duration sits beside the new fit's progress
    // and reads as the one in flight.
    setup({
      result,
      running: true,
      partial: { done: 2, total: 60, phase: "Fitting", loss: 1 },
    });
    expect(screen.queryByTestId("model-ready")).toBeNull();
  });

  it("renders a fit error in this band", () => {
    setup({ error: "No usable feature columns." });
    expect(screen.getByRole("alert")).toHaveTextContent(/no usable feature columns/i);
  });
});
