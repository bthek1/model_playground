import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { LinkMetrics } from "@/webgpu/linkPredictor";
import type { LinkSummary, LinkTrainResult } from "@/webgpu/linkSession";

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

/** 4 papers in a path, one citation held out. */
function summary(over: Partial<LinkSummary> = {}): LinkSummary {
  return {
    nNodes: 4,
    nEdges: 4,
    nTrainEdges: 2,
    nValEdges: 0,
    nTestEdges: 1,
    rescued: 0,
    rowPtr: Uint32Array.from([0, 1, 2, 3, 4]),
    colIdx: Uint32Array.from([1, 0, 3, 2]),
    labels: Uint8Array.from([0, 1, 2, 3]),
    x: Float32Array.from([0, 0.3, 0.6, 1]),
    y: Float32Array.from([0, 0.5, 0.5, 1]),
    layoutMs: 912,
    backend: "webgpu",
    ...over,
  };
}

function metric(over: Partial<LinkMetrics> = {}): LinkMetrics {
  return {
    epoch: 41,
    totalEpochs: 150,
    loss: 0.42,
    trainAuc: 0.95,
    valAuc: 0.913,
    testAuc: 0.902,
    testAp: 0.889,
    ...over,
  };
}

function runResult(over: Partial<LinkTrainResult> = {}): LinkTrainResult {
  return {
    metrics: metric(),
    candidates: Uint32Array.from([0, 2, 1, 3, 0, 3]),
    candidateScores: Float32Array.from([3.2, 2.1, 1.4]),
    embedding: Float32Array.from([1, 0, 0, 1, 1, 1, 0.5, 0.5]),
    embeddingDim: 2,
    backend: "webgpu",
    elapsedMs: 2400,
    ...over,
  };
}

const hookState = {
  status: "idle" as string,
  summary: null as LinkSummary | null,
  loadError: null as string | null,
  backend: null as string | null,
  load: vi.fn(),
  retry: vi.fn(),
  training: false,
  metrics: [] as LinkMetrics[],
  result: null as LinkTrainResult | null,
  elapsedMs: null as number | null,
  trainError: null as string | null,
  history: [] as unknown[],
  start: vi.fn(),
  stop: vi.fn(),
};
vi.mock("@/hooks/useLinkPrediction", () => ({
  useLinkPrediction: () => hookState,
}));

const { Route } = await import("@/routes/link-prediction");
const LinkPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!LinkPage) throw new Error("Link prediction route component not found");
  render(<LinkPage />);
}

function ready() {
  Object.assign(hookState, {
    status: "ready",
    summary: summary(),
    backend: "webgpu",
  });
}

function trained() {
  ready();
  Object.assign(hookState, {
    metrics: [metric()],
    result: runResult(),
    elapsedMs: 2400,
  });
}

describe("LinkPredictionPage", () => {
  beforeEach(() => {
    Object.assign(hookState, {
      status: "idle",
      summary: null,
      loadError: null,
      backend: null,
      training: false,
      metrics: [],
      result: null,
      elapsedMs: null,
      trainError: null,
      history: [],
    });
    hookState.load.mockClear();
    hookState.retry.mockClear();
    hookState.start.mockClear();
    hookState.stop.mockClear();
  });

  it("renders all four slots, with the output empty before any run", () => {
    renderPage();
    for (const slot of [1, 2, 3, 4]) {
      expect(screen.getByTestId(`slot-${slot}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("splits nothing on mount — the graph loads only when asked", () => {
    renderPage();
    expect(hookState.load).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /load graph/i })).toBeVisible();
  });

  it("says the dataset is bundled, so the wait is the split and the layout", () => {
    renderPage();
    expect(screen.getByText(/161 KB, bundled with the app/i)).toBeVisible();
    expect(screen.getByText(/no download/i)).toBeVisible();
  });

  it("disables Train until the graph is ready, while the controls stay live", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^train$/i })).toBeDisabled();
    // SELECT is a choice and commits to nothing, so it is never gated on ready.
    expect(screen.getByRole("button", { name: /graphsage/i })).toBeEnabled();
  });

  it("states that the held-out edges leave the graph, not just the loss", () => {
    // The page's one real correctness claim, and the thing a reader has to know
    // to trust the number beside it.
    renderPage();
    expect(
      screen.getByText(/removed from the graph before training, not just/i),
    ).toBeVisible();
  });

  it("changing the architecture runs nothing by itself", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /gin/i }));
    expect(hookState.start).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^train$/i }));
    expect(hookState.start).toHaveBeenCalledTimes(1);
  });

  it("re-splits when the held-out fraction changes, because it is a new graph", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "20%" }));
    expect(hookState.load).toHaveBeenCalledWith({ testFrac: 0.2 });
    // …and it does not start a run on its own.
    expect(hookState.start).not.toHaveBeenCalled();
  });

  it("does not load on a fraction change before the first load", () => {
    // Nothing downloads or lays out until the user asks. Picking a fraction
    // while idle is a choice, and choices commit to nothing.
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "30%" }));
    expect(hookState.load).not.toHaveBeenCalled();
  });

  it("reports the split and which backend a run will use", () => {
    ready();
    renderPage();
    const note = screen.getByTestId("model-ready");
    expect(note).toHaveTextContent("2 citations kept");
    expect(note).toHaveTextContent("1 test");
    expect(note).toHaveTextContent("the GPU");
  });

  it("draws nothing in OUTPUT until a run has finished", () => {
    ready();
    renderPage();
    expect(screen.queryByTestId("graph-canvas")).not.toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("draws the graph and the scores once trained", () => {
    trained();
    renderPage();
    expect(screen.getByTestId("graph-canvas")).toBeInTheDocument();
    expect(screen.queryByTestId("output-empty")).not.toBeInTheDocument();
    expect(screen.getByText("Test AUC")).toBeInTheDocument();
    expect(screen.getByText("0.902")).toBeInTheDocument();
    expect(screen.getByText("0.889")).toBeInTheDocument();
  });

  it("says the dashed lines are predictions rather than data", () => {
    // Twice, deliberately: the header frames the whole page that way, and
    // OUTPUT repeats it beside the drawing, where someone reading a line across
    // a gap needs to know it is a claim and not a citation.
    trained();
    renderPage();
    expect(
      screen.getByText(/the dashed lines are\s+predictions, not data/i),
    ).toBeVisible();
    expect(
      screen.getByText(/pairs it scores highest among those that are not/i),
    ).toBeVisible();
  });

  it("moving the top-k slider re-draws without re-running", () => {
    // The page-pattern rule: a control that only re-reads a result in hand
    // spends nothing, so it needs no press — and must not start an inference.
    trained();
    renderPage();
    fireEvent.change(screen.getByLabelText(/predicted links drawn/i), {
      target: { value: "5" },
    });
    expect(hookState.start).not.toHaveBeenCalled();
    expect(hookState.load).not.toHaveBeenCalled();
    expect(screen.getByText(/top 5/)).toBeVisible();
  });

  it("lists the top pairs as text, because the drawn ones are a few pixels long", () => {
    // The pairs a link predictor ranks highest already share neighbours, so the
    // layout has put them on top of each other. The list is the readable form of
    // the same answer, and clicking a row rings the pair on the canvas.
    trained();
    renderPage();
    const row = screen.getByRole("button", { name: /#0.*#2/ });
    expect(row).toBeVisible();

    fireEvent.click(row);
    expect(screen.getByTestId("pair-score")).toHaveTextContent("#0");
    expect(screen.getByTestId("pair-score")).toHaveTextContent("#2");
    // Reading a result in hand: no re-run, no re-split.
    expect(hookState.start).not.toHaveBeenCalled();
    expect(hookState.load).not.toHaveBeenCalled();
  });

  it("invites a pair to be picked before one has been", () => {
    trained();
    renderPage();
    expect(screen.getByText(/click a paper, then another/i)).toBeVisible();
    expect(screen.queryByTestId("pair-score")).not.toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, with a retry", () => {
    Object.assign(hookState, { status: "error", loadError: "decode failed" });
    renderPage();
    const slot = screen.getByTestId("slot-2");
    expect(slot).toHaveTextContent("decode failed");
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
