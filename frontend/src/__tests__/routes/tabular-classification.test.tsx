import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { parseCsv } from "@/tabular/csv";
import type { UseDatasetResult } from "@/hooks/useDataset";
import type { UseTabularFitResult } from "@/hooks/useTabularFit";
import type { ClassificationMetrics, Dataset, FitResult } from "@/tabular/types";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation((path: string) => (opts: Record<string, unknown>) => ({
        path,
        options: opts,
      })),
  };
});

// ECharts needs a real layout engine; jsdom/happy-dom has none, and the bars
// are asserted in the E2E suite instead.
vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const DATASET: Dataset = parseCsv(
  "age,city,defaulted\n30,north,yes\n40,south,no\n50,north,no\n60,south,yes\n",
  { name: "tiny.csv" },
).dataset;

const metrics: ClassificationMetrics = {
  accuracy: 0.8,
  precision: 0.75,
  recall: 0.7,
  f1: 0.72,
  confusion: { counts: Int32Array.from([3, 1, 1, 5]), labels: ["yes", "no"] },
  baselineAccuracy: 0.55,
  baselineLabel: "no",
};

const RESULT: FitResult = {
  spec: {
    family: "boosting",
    objective: "classification",
    targetIndex: 2,
    featureIndices: [0, 1],
    hp: {
      maxDepth: 4,
      minLeaf: 5,
      nTrees: 120,
      featureFraction: 0.7,
      shrinkage: 0.1,
      epochs: 20,
      learningRate: 0.2,
      batchSize: 64,
      hidden: 32,
    },
    seed: 42,
    testFraction: 0.25,
  },
  trainRows: 3,
  testRows: 10,
  fitMs: 1234,
  compute: "cpu",
  classification: metrics,
  probabilities: Float32Array.from([
    0.9, 0.1, 0.2, 0.8, 0.4, 0.6, 0.55, 0.45, 0.3, 0.7, 0.8, 0.2, 0.1, 0.9, 0.6,
    0.4, 0.35, 0.65, 0.7, 0.3,
  ]),
  testLabels: Uint8Array.from([0, 1, 1, 0, 1, 0, 1, 1, 1, 0]),
  trainLabels: Uint8Array.from([0, 1, 1]),
  labels: ["yes", "no"],
  importance: [
    { name: "age", drop: 0.2 },
    { name: "city", drop: 0.01 },
  ],
  curve: [{ step: 1, loss: 0.6 }],
};

const fitMock = vi.fn().mockResolvedValue(RESULT);
const predictMock = vi.fn().mockResolvedValue({
  scores: [0.3, 0.7],
  labels: ["yes", "no"],
  objective: "classification",
});
const loadMock = vi.fn();
const stopMock = vi.fn();

let fitState: UseTabularFitResult;
let dataState: UseDatasetResult;

const loadSample = vi.fn().mockResolvedValue(undefined);
const loadFile = vi.fn().mockResolvedValue(undefined);

// The argument is recorded, not read: the assertion that matters is *what the
// route passed*, and specifically that it passed no auto-load flag.
const useTabularFit = vi.fn((dataset: Dataset | null) => {
  void dataset;
  return fitState;
});
const useDataset = vi.fn(() => dataState);

vi.mock("@/hooks/useTabularFit", () => ({
  useTabularFit: (dataset: Dataset | null) => useTabularFit(dataset),
}));
vi.mock("@/hooks/useDataset", () => ({
  useDataset: () => useDataset(),
}));

const { Route } = await import("@/routes/tabular-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Tabular classification route component not found");
  render(<Page />);
}

function baseFit(extra: Partial<UseTabularFitResult> = {}): UseTabularFitResult {
  return {
    status: "idle",
    idle: true,
    loading: false,
    ready: false,
    progress: null,
    loadProgress: null,
    loadedInMs: null,
    backend: null,
    running: false,
    partial: null,
    result: null,
    error: null,
    load: loadMock,
    retry: vi.fn(),
    cancel: vi.fn(),
    fit: fitMock,
    predict: predictMock,
    stop: stopMock,
    ...extra,
  };
}

function baseData(extra: Partial<UseDatasetResult> = {}): UseDatasetResult {
  return {
    dataset: null,
    sample: null,
    parsing: false,
    error: null,
    issues: [],
    loadSample,
    loadFile,
    clear: vi.fn(),
    ...extra,
  };
}

describe("TabularClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fitState = baseFit();
    dataState = baseData();
  });

  it("renders the four slots with an empty OUTPUT before any fit", () => {
    renderPage();
    for (const step of [1, 2, 3, 4]) {
      expect(screen.getByTestId(`slot-${step}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-panel")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("fits nothing on arrival", () => {
    renderPage();
    // Both halves, as the page-pattern checklist requires: the hook is handed
    // no auto-load argument at all, *and* nothing has called load() or fit()
    // behind the user's back.
    expect(useTabularFit).toHaveBeenCalledWith(null);
    expect(loadMock).not.toHaveBeenCalled();
    expect(fitMock).not.toHaveBeenCalled();
  });

  it("states the privacy claim before anything is chosen", () => {
    renderPage();
    expect(screen.getByText(/never leaves this device/i)).toBeInTheDocument();
  });

  it("choosing a dataset fits nothing", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /palmer penguins/i }));
    expect(loadSample).toHaveBeenCalledOnce();
    expect(fitMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("picking a target column and a feature fits nothing, and then FIT does", async () => {
    // The single most valuable assertion on this page: "pick a target column
    // and it fits" feels responsive and is the five-samples-five-inferences
    // failure with a dropdown in front of it.
    dataState = baseData({ dataset: DATASET });
    renderPage();

    fireEvent.change(screen.getByLabelText(/target/i), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: /^age/i }));
    fireEvent.change(screen.getByLabelText(/held out for testing/i), {
      target: { value: "0.4" },
    });
    fireEvent.change(screen.getByLabelText(/trees/i), { target: { value: "80" } });
    expect(fitMock).not.toHaveBeenCalled();
    expect(loadMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("fit-button"));
    await waitFor(() => expect(fitMock).toHaveBeenCalledOnce());
  });

  it("changing the family swaps in that family's own defaults", () => {
    // Inheriting one family's hyperparameters into another is the
    // /link-prediction mistake — reuse that looks like a decision.
    dataState = baseData({ dataset: DATASET });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /random forest/i }));
    expect(screen.getByLabelText(/max depth/i)).toHaveValue("8");
    fireEvent.click(screen.getByRole("button", { name: /gradient boosting/i }));
    expect(screen.getByLabelText(/max depth/i)).toHaveValue("4");
    expect(fitMock).not.toHaveBeenCalled();
  });

  it("says where each family's arithmetic runs", () => {
    dataState = baseData({ dataset: DATASET });
    renderPage();
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/your CPU/i);
    fireEvent.click(screen.getByRole("button", { name: /logistic regression/i }));
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/your GPU/i);
  });

  it("caps boosting's depth lower than the forest's, from the measurement", () => {
    dataState = baseData({ dataset: DATASET });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /gradient boosting/i }));
    expect(screen.getByLabelText(/max depth/i)).toHaveAttribute("max", "6");
    fireEvent.click(screen.getByRole("button", { name: /random forest/i }));
    expect(screen.getByLabelText(/max depth/i)).toHaveAttribute("max", "12");
  });

  it("blocks FIT with a reason until there is something to fit", () => {
    renderPage();
    expect(screen.getByTestId("fit-button")).toBeDisabled();
    expect(screen.getByText(/choose a sample or drop a csv/i)).toBeInTheDocument();
  });

  it("leaves the input sources usable while the trigger is gated", () => {
    // The corollary of "only FIT and PREDICT spend": the sources are not gated
    // on readiness, only the trigger is.
    renderPage();
    expect(screen.getByTestId("predict-button")).toBeDisabled();
    expect(screen.getByRole("button", { name: /wine recognition/i })).toBeEnabled();
    expect(screen.getByLabelText(/csv file/i)).toBeEnabled();
  });

  it("renders the metric block, its baseline and the matrix after a fit", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({ status: "ready", idle: false, ready: true, result: RESULT });
    renderPage();
    expect(screen.getByTestId("metric-block")).toBeInTheDocument();
    // 60%, not the 55% the fixture's own metrics object carries: on a binary
    // target the page re-derives the whole block from the held-out
    // probabilities at the current threshold, so what is on screen is computed
    // from `testLabels`/`trainLabels` and cannot drift from the matrix beside it.
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(/60\.0%/);
    expect(screen.getByTestId("confusion-matrix")).toBeInTheDocument();
    expect(screen.getByTestId("importance-bars")).toBeInTheDocument();
    expect(screen.queryByTestId("output-empty")).not.toBeInTheDocument();
  });

  it("moves the threshold without touching the worker", () => {
    // The legitimate re-derivation: it re-reads the held-out probabilities the
    // fit already returned, on the main thread. A slider that refits would be a
    // ten-second wait behind a control that looks free.
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({ status: "ready", idle: false, ready: true, result: RESULT });
    renderPage();
    const before = within(screen.getByTestId("confusion-matrix")).getAllByRole("cell")
      .map((c) => c.textContent)
      .join(",");
    fireEvent.change(screen.getByLabelText(/decision threshold/i), {
      target: { value: "0.2" },
    });
    const after = within(screen.getByTestId("confusion-matrix")).getAllByRole("cell")
      .map((c) => c.textContent)
      .join(",");
    expect(after).not.toBe(before);
    expect(fitMock).not.toHaveBeenCalled();
    expect(predictMock).not.toHaveBeenCalled();
    expect(loadMock).not.toHaveBeenCalled();
  });

  it("predicts a typed row only when PREDICT is pressed", async () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({ status: "ready", idle: false, ready: true, result: RESULT });
    renderPage();
    const boxes = screen.getAllByRole("textbox");
    fireEvent.change(boxes[0], { target: { value: "35" } });
    expect(predictMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(predictMock).toHaveBeenCalledOnce());
    expect(await screen.findByTestId("prediction")).toHaveTextContent(/70\.0%/);
  });

  it("shows the determinate counter and a Stop while fitting", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({
      status: "ready",
      idle: false,
      ready: true,
      running: true,
      partial: { done: 30, total: 120, phase: "Fitting", loss: 0.42 },
    });
    renderPage();
    expect(screen.getByTestId("fit-progress")).toHaveTextContent("30/120");
    fireEvent.click(screen.getByTestId("fit-stop"));
    expect(stopMock).toHaveBeenCalledOnce();
  });

  it("puts each error in the slot that produced it", () => {
    dataState = baseData({ dataset: DATASET, error: "Could not read that file." });
    fitState = baseFit({ status: "error", idle: false, error: "The fit failed." });
    renderPage();
    // A load-side failure belongs to the FIT band…
    expect(within(screen.getByTestId("slot-2")).getByTestId("error-note")).toHaveTextContent(
      /fit failed/i,
    );
    // …and an input-side one to RUN.
    expect(within(screen.getByTestId("slot-3")).getByTestId("error-note")).toHaveTextContent(
      /could not read/i,
    );
  });

  it("reports the sampled-row count rather than quietly modelling a subset", () => {
    dataState = baseData({
      dataset: { ...DATASET, sourceRowCount: 900_000, sampled: true },
    });
    renderPage();
    expect(screen.getByTestId("dataset-summary")).toHaveTextContent(/900,000 rows/);
  });

  it("names the skipped rows and where the first one is", () => {
    dataState = baseData({
      dataset: DATASET,
      issues: [{ line: 42, message: "3 fields, expected 4" }],
    });
    renderPage();
    expect(screen.getByTestId("dataset-summary")).toHaveTextContent(/line 42/);
  });
});
