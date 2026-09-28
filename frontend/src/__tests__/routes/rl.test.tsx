// `/rl` — the page-pattern §8 checklist with the fitting-page substitutions:
// nothing trains on mount, choosing anything trains nothing, and the band that
// is missing says why it is missing. Layout is not asserted here (jsdom has no
// geometry); `e2e/specs/rl.spec.ts` does that.

import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RlRunRecord } from "@/hooks/useRlTraining";
import type { QLearningRequest, RlRenderState, RlTrainResult } from "@/rl/types";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation((path: string) => (opts: Record<string, unknown>) => ({ path, options: opts })),
  };
});

vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const hookState = {
  training: false,
  returns: [] as number[],
  progress: null as null | { episode: number; totalEpisodes: number; steps: number; epsilon: number; render: RlRenderState },
  render: null as RlRenderState | null,
  result: null as RlTrainResult | null,
  error: null as string | null,
  history: [] as RlRunRecord[],
  start: vi.fn(),
  stop: vi.fn(),
  control: vi.fn(),
  clearHistory: vi.fn(),
};
const useRlTraining = vi.fn<(...args: unknown[]) => typeof hookState>(() => hookState);
vi.mock("@/hooks/useRlTraining", () => ({
  useRlTraining: (...a: unknown[]) => useRlTraining(...a),
  runKey: (r: unknown) => JSON.stringify(r),
}));

const { Route, DEFAULTS } = await import("@/routes/rl");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("RL route component not found");
  return render(<Page />);
}

/** A table whose greedy policy walks the deterministic 4×4's optimal path. */
function solvedTable(): Float32Array {
  const q = new Float32Array(64);
  // down, down, right, down, right, right: 0 → 4 → 8 → 9 → 13 → 14 → 15
  const path: [number, number][] = [[0, 1], [4, 1], [8, 2], [9, 1], [13, 2], [14, 2]];
  path.forEach(([s, a], i) => (q[s * 4 + a] = 0.95 ** (path.length - 1 - i)));
  return q;
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.assign(hookState, {
    training: false,
    returns: [],
    progress: null,
    render: null,
    result: null,
    error: null,
    history: [],
  });
});

describe("RlPage", () => {
  it("renders three bands and says why the load band is missing", () => {
    renderPage();
    expect(screen.getByTestId("slot-1")).toBeInTheDocument();
    expect(screen.getByTestId("slot-2")).toBeInTheDocument();
    expect(screen.getByTestId("slot-3")).toBeInTheDocument();
    expect(screen.queryByTestId("slot-4")).toBeNull();
    const band = screen.getByTestId("no-load-band");
    expect(band).toHaveTextContent(/nothing to load/i);
    expect(band).toHaveTextContent(/the CPU, on purpose/i);
    expect(band).toHaveTextContent(/no ONNX export/i);
  });

  it("trains nothing on mount", () => {
    renderPage();
    expect(useRlTraining).toHaveBeenCalledWith(); // no autoLoad, no arguments at all
    expect(hookState.start).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("trains nothing when the grid, a hyperparameter or the seed changes — only Train does", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /8 × 8/ }));
    fireEvent.click(screen.getByRole("button", { name: /not slippery/i }));
    fireEvent.change(screen.getByLabelText(/exploration ε/i), { target: { value: "0.3" } });
    fireEvent.change(screen.getByLabelText(/discount γ/i), { target: { value: "0.9" } });
    fireEvent.change(screen.getByLabelText(/^seed$/i), { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: /3,000 steps\/s/ }));
    expect(hookState.start).not.toHaveBeenCalled();
    expect(hookState.control).not.toHaveBeenCalled(); // no run to steer
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("train-button"));
    expect(hookState.start).toHaveBeenCalledTimes(1);
    const [req, speed] = hookState.start.mock.calls[0] as [QLearningRequest, number | null];
    expect(req).toMatchObject({ map: "8x8", slippery: true, epsilon: 0.3, gamma: 0.9, seed: 7 });
    expect(speed).toBe(3000);
  });

  it("swaps in the environment's own defaults rather than inheriting the last one's", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /not slippery/i })); // → slippery 4×4
    fireEvent.click(screen.getByTestId("train-button"));
    const [req] = hookState.start.mock.calls[0] as [QLearningRequest];
    expect(req.alpha).toBe(DEFAULTS["4x4-slip"].alpha);
    expect(req.episodes).toBe(DEFAULTS["4x4-slip"].episodes);
    expect(DEFAULTS["4x4-slip"].alpha).not.toBe(DEFAULTS["4x4-det"].alpha);
  });

  it("keeps Train and Stop in the RUN transport, and Stop live only while training", () => {
    renderPage();
    const run = screen.getByTestId("slot-2");
    expect(within(run).getByTestId("train-button")).toBeEnabled();
    expect(within(run).getByRole("button", { name: /stop/i })).toBeDisabled();
  });

  it("steers a run in progress with ε and speed, without starting another", () => {
    hookState.training = true;
    renderPage();
    const live = screen.getByLabelText(/live ε/i);
    expect(live).toBeEnabled();
    fireEvent.change(live, { target: { value: "0" } });
    expect(hookState.control).toHaveBeenCalledWith({ epsilon: 0 });
    fireEvent.click(screen.getByRole("button", { name: /^max$/ }));
    expect(hookState.control).toHaveBeenCalledWith({ speed: null });
    fireEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(hookState.stop).toHaveBeenCalled();
    expect(hookState.start).not.toHaveBeenCalled();
    expect(screen.getByTestId("train-button")).toBeDisabled();
  });

  it("leaves the live ε disabled when there is no run to steer", () => {
    renderPage();
    expect(screen.getByLabelText(/live ε/i)).toBeDisabled();
  });

  it("states the ε = 0 lesson beside the control", () => {
    renderPage();
    expect(screen.getByTestId("epsilon-lesson")).toHaveTextContent(
      /set ε to 0 and the agent never finds the goal/i,
    );
  });

  it("puts a training error in the RUN band", () => {
    hookState.error = "bad grid";
    renderPage();
    expect(within(screen.getByTestId("slot-2")).getByTestId("error-note")).toHaveTextContent("bad grid");
  });

  it("scores a finished table against value iteration's exact policy", () => {
    const render: RlRenderState = { kind: "grid", map: "4x4", q: solvedTable(), agent: 15 };
    Object.assign(hookState, {
      render,
      returns: [0, 1, 1, 1],
      result: { episodes: 4, steps: 30, elapsedMs: 3, stopped: false, epsilonChanged: false, render },
      history: [],
    });
    renderPage();
    expect(screen.queryByTestId("output-empty")).toBeNull();
    expect(screen.getByTestId("success-rate")).toHaveTextContent("75%");
    // The path cells are optimal; the off-path cells still point LEFT by tie
    // and are not all optimal — so this is a real count, not "all of them".
    const agreement = screen.getByTestId("policy-agreement").textContent ?? "";
    const [agree, cells] = agreement.match(/\d+/g)!.map(Number);
    expect(cells).toBe(11);
    expect(agree).toBeGreaterThanOrEqual(6);
    expect(agree).toBeLessThan(11);
    expect(screen.getByTestId("greedy-rate")).toHaveTextContent("100%");
    // The policy as text: the start cell's arrow reads down.
    const start = screen.getByTestId("policy-grid").querySelector('[data-state="0"]');
    expect(start).toHaveAttribute("data-action", "down");
  });

  it("writes nothing to storage and fetches nothing", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /8 × 8/ }));
    fireEvent.click(screen.getByTestId("train-button"));
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
    setItem.mockRestore();
  });
});
