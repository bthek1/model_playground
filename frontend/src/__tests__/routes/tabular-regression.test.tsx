import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseDatasetResult } from "@/hooks/useDataset";
import type { UseTabularFitResult } from "@/hooks/useTabularFit";
import { parseCsv } from "@/tabular/csv";
import type { Dataset, FitResult, RegressionMetrics } from "@/tabular/types";

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

vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const DATASET: Dataset = parseCsv(
  "age,city,price\n30,north,10\n40,south,20\n50,north,30\n60,south,40\n",
  { name: "tiny.csv" },
).dataset;

const HP = {
  maxDepth: 4,
  minLeaf: 5,
  nTrees: 150,
  featureFraction: 0.7,
  shrinkage: 0.08,
  epochs: 20,
  learningRate: 0.2,
  batchSize: 64,
  hidden: 32,
  lambda: 1,
};

const metrics: RegressionMetrics = {
  rmse: 2.5,
  mae: 1.9,
  r2: 0.82,
  baselineRmse: 6.1,
  baselineMae: 5.2,
  units: "price",
};

function makeResult(extra: Partial<FitResult> = {}): FitResult {
  return {
    spec: {
      family: "boosting",
      objective: "regression",
      targetIndex: 2,
      featureIndices: [0, 1],
      hp: HP,
      seed: 42,
      testFraction: 0.25,
    },
    trainRows: 3,
    testRows: 4,
    droppedRows: 0,
    fitMs: 900,
    compute: "cpu",
    regression: metrics,
    predictions: Float32Array.from([9, 21, 29, 41]),
    actuals: Float32Array.from([10, 20, 30, 40]),
    importance: [{ name: "age", drop: 0.4 }],
    curve: [],
    ...extra,
  };
}

const fitMock = vi.fn().mockResolvedValue(makeResult());
const predictMock = vi.fn().mockResolvedValue({
  scores: [31.5],
  labels: [],
  objective: "regression",
});
const loadMock = vi.fn();
const stopMock = vi.fn();

let fitState: UseTabularFitResult;
let dataState: UseDatasetResult;

const useTabularFit = vi.fn((dataset: Dataset | null) => {
  void dataset;
  return fitState;
});
const useDataset = vi.fn(() => dataState);

vi.mock("@/hooks/useTabularFit", () => ({
  useTabularFit: (dataset: Dataset | null) => useTabularFit(dataset),
}));
vi.mock("@/hooks/useDataset", () => ({ useDataset: () => useDataset() }));

const { Route } = await import("@/routes/tabular-regression");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Tabular regression route component not found");
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
    loadSample: vi.fn().mockResolvedValue(undefined),
    loadFile: vi.fn().mockResolvedValue(undefined),
    clear: vi.fn(),
    ...extra,
  };
}

describe("TabularRegressionPage", () => {
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
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("fits nothing on arrival", () => {
    renderPage();
    expect(useTabularFit).toHaveBeenCalledWith(null);
    expect(loadMock).not.toHaveBeenCalled();
    expect(fitMock).not.toHaveBeenCalled();
  });

  it("offers the regression ladder, not the classification one", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /ridge regression/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /quantile regression/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /logistic regression/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /neural network/i })).toBeNull();
  });

  it("says ridge's arithmetic is split across the GPU and the CPU", () => {
    dataState = baseData({ dataset: DATASET });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /ridge regression/i }));
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/your GPU/i);
    expect(screen.getByTestId("family-compute")).toHaveTextContent(/factorisation is microseconds/i);
  });

  it("offers only numeric columns as a target", () => {
    dataState = baseData({ dataset: DATASET });
    renderPage();
    const options = within(screen.getByLabelText(/target/i)).getAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual(["age", "price"]);
  });

  it("flipping the log toggle fits nothing, and the next FIT does", async () => {
    // It cannot re-derive — a fit on log1p(y) is a different fit — so the
    // toggle spends, and the page says so before the click rather than after.
    dataState = baseData({ dataset: DATASET });
    renderPage();
    fireEvent.click(screen.getByLabelText(/fit on log/i));
    expect(fitMock).not.toHaveBeenCalled();
    expect(loadMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("log-toggle")).toHaveTextContent(/real second fit/i);

    fireEvent.click(screen.getByTestId("fit-button"));
    await waitFor(() => expect(fitMock).toHaveBeenCalledOnce());
    expect(fitMock.mock.calls[0][0]).toMatchObject({ logTarget: true });
  });

  it("asks for quantiles only when the quantile family is chosen", async () => {
    dataState = baseData({ dataset: DATASET });
    renderPage();
    fireEvent.click(screen.getByTestId("fit-button"));
    await waitFor(() => expect(fitMock).toHaveBeenCalled());
    expect(fitMock.mock.calls[0][0].quantiles).toBeUndefined();

    fitMock.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /quantile regression/i }));
    fireEvent.click(screen.getByTestId("fit-button"));
    await waitFor(() => expect(fitMock).toHaveBeenCalled());
    expect(fitMock.mock.calls[0][0].quantiles).toEqual([0.1, 0.5, 0.9]);
  });

  it("renders the metric block with its baseline, both plots and the importances", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({ status: "ready", idle: false, ready: true, result: makeResult() });
    renderPage();
    expect(screen.getByTestId("regression-metrics")).toBeInTheDocument();
    expect(screen.getByTestId("regression-baseline")).toHaveTextContent(/6\.10 price/);
    expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument();
    expect(screen.getByTestId("residual-plot")).toBeInTheDocument();
    expect(screen.getByTestId("importance-bars")).toBeInTheDocument();
    expect(screen.queryByTestId("output-empty")).not.toBeInTheDocument();
  });

  it("keeps the log-space numbers in their own block, with their units named", () => {
    // The whole point of the toggle. Rendering these inline beside the numbers
    // above *is* the mistake being demonstrated.
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({
      status: "ready",
      idle: false,
      ready: true,
      result: makeResult({
        logSpaceRegression: { ...metrics, rmse: 0.04, mae: 0.03, units: "log(1 + price)" },
      }),
    });
    renderPage();
    const block = screen.getByTestId("log-space-metrics");
    expect(block).toHaveTextContent(/log\(1 \+ price\)/);
    expect(block).toHaveTextContent(/not comparable/i);
    expect(screen.getByTestId("regression-metrics")).toHaveTextContent(/price/);
  });

  it("shows ridge's coefficients instead of permutation bars", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({
      status: "ready",
      idle: false,
      ready: true,
      result: makeResult({
        spec: { ...makeResult().spec, family: "ridge" },
        coefficients: [
          { name: "age", weight: 2.5 },
          { name: "city = north", weight: -0.4 },
        ],
      }),
    });
    renderPage();
    expect(screen.getByTestId("coefficients")).toHaveTextContent("age");
    expect(screen.queryByTestId("importance-bars")).toBeNull();
  });

  it("says so when the design is rank-deficient", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({
      status: "ready",
      idle: false,
      ready: true,
      result: makeResult({ rankDeficient: true, coefficients: [{ name: "age", weight: 1 }] }),
    });
    renderPage();
    expect(screen.getByTestId("rank-deficient")).toHaveTextContent(/no unique solution/i);
  });

  it("reports the band's measured coverage, not merely that a band exists", () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({
      status: "ready",
      idle: false,
      ready: true,
      result: makeResult({
        spec: { ...makeResult().spec, family: "quantile", quantiles: [0.1, 0.5, 0.9] },
        quantilePredictions: new Float32Array(12),
        quantileCoverage: 0.78,
      }),
    });
    renderPage();
    expect(screen.getByTestId("band-coverage")).toHaveTextContent("78.0%");
  });

  it("predicts a typed row only when PREDICT is pressed", async () => {
    dataState = baseData({ dataset: DATASET });
    fitState = baseFit({ status: "ready", idle: false, ready: true, result: makeResult() });
    renderPage();
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "45" } });
    expect(predictMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("predict-button"));
    await waitFor(() => expect(predictMock).toHaveBeenCalledOnce());
    expect(await screen.findByTestId("prediction")).toHaveTextContent("31.500");
  });

  it("puts each error in the slot that produced it", () => {
    dataState = baseData({ dataset: DATASET, error: "Could not read that file." });
    fitState = baseFit({ status: "error", idle: false, error: "The fit failed." });
    renderPage();
    expect(within(screen.getByTestId("slot-2")).getByTestId("error-note")).toHaveTextContent(
      /fit failed/i,
    );
    expect(within(screen.getByTestId("slot-3")).getByTestId("error-note")).toHaveTextContent(
      /could not read/i,
    );
  });
});
