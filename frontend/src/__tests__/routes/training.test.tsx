import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TrainMetrics, TrainResult } from "@/webgpu/linearModel";
import type { WeightSnapshot } from "@/webgpu/workerClient";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({ path, options: opts }),
      ),
  };
});

// Stub the lazily-imported chart so we don't pull echarts into the test.
vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const gpuState = { supported: true, loading: false, capabilities: null };
vi.mock("@/hooks/useWebGPU", () => ({ useWebGPU: () => gpuState }));

const hookState = {
  datasetStatus: "idle" as string,
  datasetProgress: 0,
  datasetError: null as string | null,
  poolCount: 0,
  loadData: vi.fn(),
  training: false,
  metrics: [] as TrainMetrics[],
  result: null as TrainResult | null,
  snapshot: null as WeightSnapshot | null,
  trainError: null as string | null,
  start: vi.fn(),
  stop: vi.fn(),
};
vi.mock("@/hooks/useLinearTraining", () => ({
  useLinearTraining: () => hookState,
}));

const { Route } = await import("@/routes/training");
const TrainingPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!TrainingPage) throw new Error("Training route component not found");
  render(<TrainingPage />);
}

const DEFAULTS = {
  learningRate: 0.5,
  batchSize: 64,
  epochs: 10,
  trainSize: 8000,
  testSize: 2000,
};

describe("TrainingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    gpuState.supported = true;
    gpuState.loading = false;
    Object.assign(hookState, {
      datasetStatus: "idle",
      datasetProgress: 0,
      datasetError: null,
      poolCount: 0,
      training: false,
      metrics: [],
      result: null,
      snapshot: null,
      trainError: null,
    });
  });

  it("renders the page heading", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /linear model training/i }),
    ).toBeInTheDocument();
  });

  it("warns when WebGPU is unavailable", () => {
    gpuState.supported = false;
    renderPage();
    expect(screen.getByText(/webgpu isn't available/i)).toBeInTheDocument();
  });

  it("does not warn when WebGPU is ready", () => {
    renderPage();
    expect(screen.queryByText(/webgpu isn't available/i)).not.toBeInTheDocument();
  });

  it("loads the dataset for the configured train+test size", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /load mnist/i }));
    expect(hookState.loadData).toHaveBeenCalledWith(
      DEFAULTS.trainSize + DEFAULTS.testSize,
    );
  });

  it("keeps Start disabled until the dataset is ready", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /start training/i }),
    ).toBeDisabled();
  });

  it("dispatches training with the current settings once ready", () => {
    hookState.datasetStatus = "ready";
    renderPage();
    const start = screen.getByRole("button", { name: /start training/i });
    expect(start).not.toBeDisabled();
    fireEvent.click(start);
    expect(hookState.start).toHaveBeenCalledWith(DEFAULTS);
  });

  it("Stop is enabled while training and calls stop()", () => {
    hookState.datasetStatus = "ready";
    hookState.training = true;
    renderPage();
    const stop = screen.getByRole("button", { name: /^stop$/i });
    expect(stop).not.toBeDisabled();
    fireEvent.click(stop);
    expect(hookState.stop).toHaveBeenCalledTimes(1);
  });

  it("renders live stats and charts once metrics arrive", async () => {
    hookState.datasetStatus = "ready";
    hookState.metrics = [
      { epoch: 0, step: 1, totalSteps: 4, loss: 1.5, trainAcc: 0.8, testAcc: 0.75 },
    ];
    renderPage();

    expect(screen.getByText("1.5000")).toBeInTheDocument(); // batch loss
    expect(screen.getByText("80.00%")).toBeInTheDocument(); // train acc
    expect(screen.getByText("75.00%")).toBeInTheDocument(); // test acc
    // Two lazy charts (loss + accuracy) resolve through Suspense.
    expect(await screen.findAllByTestId("echart")).toHaveLength(2);
  });

  it("renders the model weight templates once a snapshot arrives", () => {
    hookState.datasetStatus = "ready";
    hookState.snapshot = {
      epoch: 0,
      weights: new Float32Array(784 * 10),
      bias: new Float32Array(10),
    };
    renderPage();

    expect(screen.getByText(/the model/i)).toBeInTheDocument();
    // One labelled canvas per class (digits 0–9).
    expect(
      screen.getByLabelText(/weight template for digit 0/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/weight template for digit 9/i),
    ).toBeInTheDocument();
  });

  it("shows dataset and training errors", () => {
    hookState.datasetError = "could not load sprite";
    hookState.trainError = "shapes must match";
    renderPage();
    expect(screen.getByText(/could not load sprite/i)).toBeInTheDocument();
    expect(screen.getByText(/shapes must match/i)).toBeInTheDocument();
  });
});
