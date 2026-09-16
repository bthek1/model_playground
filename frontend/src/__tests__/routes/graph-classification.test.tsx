import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphClassMetrics } from "@/webgpu/graphPool";
import type { GraphLayoutPayload, ProteinSummary } from "@/webgpu/proteinSession";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({
          path,
          options: opts,
        }),
      ),
  };
});

vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const gpuState = {
  supported: true,
  loading: false,
  capabilities: {
    status: "ready",
    adapter: { vendor: "test", architecture: "gpu", device: "", description: "" },
    isFallbackAdapter: false,
    features: [],
    limits: {},
  },
};
vi.mock("@/hooks/useWebGPU", () => ({ useWebGPU: () => gpuState }));

/** Four graphs: two train, two test. Labels 0,0,1,0 — majority is 0. */
function summary(over: Partial<ProteinSummary> = {}): ProteinSummary {
  return {
    nGraphs: 4,
    nNodes: 10,
    nFeat: 3,
    nClasses: 2,
    nEdges: 12,
    nTrain: 2,
    nVal: 0,
    nTest: 2,
    labels: Uint8Array.from([0, 0, 1, 0]),
    graphPtr: Uint32Array.from([0, 3, 5, 8, 10]),
    testIdx: Uint32Array.from([2, 3]),
    baselineAcc: 0.5,
    fromCache: false,
    loadMs: 840,
    backend: "webgpu",
    ...over,
  };
}

function metric(over: Partial<GraphClassMetrics> = {}): GraphClassMetrics {
  return {
    epoch: 41,
    totalEpochs: 150,
    loss: 0.44,
    trainAcc: 0.81,
    valAcc: 0.74,
    testAcc: 0.735,
    baselineAcc: 0.598,
    ...over,
  };
}

const hookState = {
  status: "idle" as string,
  summary: null as ProteinSummary | null,
  loadError: null as string | null,
  backend: null as string | null,
  load: vi.fn(),
  retry: vi.fn(),
  training: false,
  metrics: [] as GraphClassMetrics[],
  predicted: null as Uint8Array | null,
  elapsedMs: null as number | null,
  trainError: null as string | null,
  history: [] as unknown[],
  start: vi.fn(),
  stop: vi.fn(),
  layouts: new Map<number, GraphLayoutPayload>(),
  requestLayout: vi.fn(),
};
vi.mock("@/hooks/useGraphClassifier", () => ({
  useGraphClassifier: () => hookState,
}));

const { Route } = await import("@/routes/graph-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Graph classification route component not found");
  render(<Page />);
}

function ready() {
  Object.assign(hookState, {
    status: "ready",
    summary: summary(),
    backend: "webgpu",
  });
}

function trained(over: Partial<GraphClassMetrics> = {}) {
  ready();
  Object.assign(hookState, {
    metrics: [metric(over)],
    predicted: Uint8Array.from([0, 0, 1, 1]),
    elapsedMs: 5200,
  });
}

describe("GraphClassificationPage", () => {
  beforeEach(() => {
    Object.assign(hookState, {
      status: "idle",
      summary: null,
      loadError: null,
      backend: null,
      training: false,
      metrics: [],
      predicted: null,
      elapsedMs: null,
      trainError: null,
      history: [],
      layouts: new Map(),
    });
    hookState.load.mockClear();
    hookState.retry.mockClear();
    hookState.start.mockClear();
    hookState.stop.mockClear();
    hookState.requestLayout.mockClear();
  });

  it("renders all four slots, with the output empty before any run", () => {
    renderPage();
    for (const slot of [1, 2, 3, 4]) {
      expect(screen.getByTestId(`slot-${slot}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("downloads nothing on mount — the dataset loads only when asked", () => {
    // The only page in the repo that fetches a dataset, so this matters more
    // here than anywhere else in the category.
    renderPage();
    expect(hookState.load).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /load dataset/i })).toBeVisible();
  });

  it("quotes the download before it happens, and says it is cached after", () => {
    renderPage();
    expect(screen.getByText(/2\.0 MB from the Hugging Face Hub/i)).toBeVisible();
    expect(screen.getByText(/one-time wait/i)).toBeVisible();
  });

  it("disables Train until the dataset is ready, while SELECT stays live", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^train$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /graphsage/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^sum$/i })).toBeEnabled();
  });

  it("changing the architecture or the readout runs nothing by itself", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /gcn/i }));
    fireEvent.click(screen.getByRole("button", { name: /^sum$/i }));
    expect(hookState.start).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^train$/i }));
    expect(hookState.start).toHaveBeenCalledTimes(1);
    expect(hookState.start.mock.calls[0]![0]).toMatchObject({
      arch: "gcn",
      readout: "sum",
    });
  });

  it("reports the dataset, where it came from, and which backend will run", () => {
    ready();
    renderPage();
    const note = screen.getByTestId("model-ready");
    expect(note).toHaveTextContent("4 graphs");
    expect(note).toHaveTextContent("downloaded");
    expect(note).toHaveTextContent("the GPU");
  });

  it("says when the dataset came out of the cache instead", () => {
    Object.assign(hookState, {
      status: "ready",
      summary: summary({ fromCache: true }),
    });
    renderPage();
    expect(screen.getByTestId("model-ready")).toHaveTextContent("read from cache");
  });

  it("draws nothing in OUTPUT until a run has produced an epoch", () => {
    ready();
    renderPage();
    expect(screen.queryByTestId("protein-gallery")).not.toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("never shows an accuracy without the baseline beside it", () => {
    // The assertion that protects this page's readability. PROTEINS is 663/450,
    // so a model that ignores the molecule scores 0.598 — an accuracy on its own
    // cannot be told apart from one.
    trained();
    renderPage();
    expect(screen.getByText("Test accuracy")).toBeInTheDocument();
    expect(screen.getByText("73.5%")).toBeInTheDocument();
    expect(screen.getByText("Majority baseline")).toBeInTheDocument();
    expect(screen.getByText("59.8%")).toBeInTheDocument();
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(
      /13\.7 points above the baseline/i,
    );
  });

  it("says plainly when the model has not beaten the baseline", () => {
    // The case the page exists to make legible: 0.60 looks like a result and is
    // not one.
    trained({ testAcc: 0.598 });
    renderPage();
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(
      /has not beaten the baseline/i,
    );
    expect(screen.getByTestId("baseline-note")).toHaveTextContent(
      /learned the class prior/i,
    );
  });

  it("draws a tile per held-out protein and asks for its layout", () => {
    trained();
    renderPage();
    expect(screen.getByTestId("protein-gallery")).toBeInTheDocument();
    // Two test graphs, neither laid out yet.
    expect(hookState.requestLayout).toHaveBeenCalledWith(2);
    expect(hookState.requestLayout).toHaveBeenCalledWith(3);
  });

  it("picking a protein re-reads the result rather than re-running", () => {
    trained();
    renderPage();
    const tiles = screen.getAllByRole("button", { name: /protein #?3/i });
    fireEvent.click(tiles[0]);

    expect(hookState.start).not.toHaveBeenCalled();
    const note = screen.getByTestId("selected-note");
    // Graph 3 is really class 0 and was predicted class 1 — a miss, and the note
    // has to say so rather than only showing the label.
    expect(note).toHaveTextContent("#3");
    expect(note).toHaveTextContent("✗");
  });

  it("says it is a single linear readout, not GIN's MLP head", () => {
    renderPage();
    expect(screen.getByText(/single linear readout/i)).toBeVisible();
  });

  it("puts a load failure in the LOAD slot, with a retry", () => {
    Object.assign(hookState, {
      status: "error",
      loadError: "the dataset could not be fetched (503)",
    });
    renderPage();
    expect(screen.getByTestId("slot-2")).toHaveTextContent("503");
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(hookState.retry).toHaveBeenCalled();
  });

  it("puts a training failure in the RUN slot", () => {
    ready();
    Object.assign(hookState, { trainError: "device lost" });
    renderPage();
    expect(screen.getByTestId("slot-3")).toHaveTextContent("device lost");
  });

  it("offers Stop only while a run is in flight", () => {
    ready();
    renderPage();
    expect(screen.getByRole("button", { name: /stop/i })).toBeDisabled();

    Object.assign(hookState, { training: true });
    renderPage();
    const stops = screen.getAllByRole("button", { name: /stop/i });
    expect(stops[stops.length - 1]).toBeEnabled();
    fireEvent.click(stops[stops.length - 1]);
    expect(hookState.stop).toHaveBeenCalled();
  });
});
