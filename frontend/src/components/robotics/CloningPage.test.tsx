import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RlRunRecord } from "@/hooks/useRlTraining";
import type { CloningRequest, RlRenderState, RlTrainResult } from "@/rl/types";

vi.mock("@/components/charts/EChart", () => ({ default: () => <div data-testid="echart" /> }));

const hookState = {
  training: false,
  returns: [] as number[],
  progress: null as null | { episode: number; totalEpisodes: number; steps: number; epsilon: null; render: RlRenderState },
  render: null as RlRenderState | null,
  result: null as RlTrainResult | null,
  error: null as string | null,
  history: [] as RlRunRecord[],
  start: vi.fn(),
  startAll: vi.fn(),
  stop: vi.fn(),
  control: vi.fn(),
  clearHistory: vi.fn(),
};
const useRlTraining = vi.fn<(...args: unknown[]) => typeof hookState>(() => hookState);
vi.mock("@/hooks/useRlTraining", () => ({ useRlTraining: (...a: unknown[]) => useRlTraining(...a) }));

const { CloningPage } = await import("./CloningPage");
const { CloningRun } = await import("@/rl/behaviourCloning");

beforeEach(() => {
  vi.clearAllMocks();
  Object.assign(hookState, { training: false, returns: [], progress: null, render: null, result: null, error: null, history: [] });
});

function renderPage() {
  return render(<CloningPage select={<div data-testid="picker" />} />);
}

/** A real, finished run's record — the cloning page reads the policy out of it. */
function record(mix: "both" | "one"): RlRunRecord {
  const request: CloningRequest = { algorithm: "behaviour-cloning", mix, demos: 20, obstacle: 0, hidden: 64, epochs: 100, lr: 0.01, seed: 1 };
  const run = new CloningRun(request);
  while (!run.done) run.step();
  const render = run.render();
  return {
    key: JSON.stringify(request),
    request,
    returns: [0.5, 0.2],
    result: { episodes: 100, steps: 1800, elapsedMs: 300, stopped: false, epsilonChanged: false, render },
  };
}

describe("CloningPage", () => {
  it("renders all four bands, trains nothing on mount, and fetches nothing", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderPage();
    for (const n of [1, 2, 3, 4]) expect(screen.getByTestId(`slot-${n}`)).toBeInTheDocument();
    expect(useRlTraining).toHaveBeenCalledWith();
    expect(hookState.start).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    expect(screen.getByTestId("model-ready")).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it("names the fix before any run, and says it does not demonstrate it", () => {
    renderPage();
    const fix = screen.getByTestId("cloning-fix");
    expect(fix).toHaveTextContent(/action chunking/i);
    expect(fix).toHaveTextContent(/diffusion policies/i);
    expect(fix).toHaveTextContent(/does not demonstrate either/i);
  });

  it("updates the preview as the set changes, and trains nothing until Train", () => {
    renderPage();
    const summary = () => screen.getByTestId("demo-summary").textContent ?? "";
    expect(summary()).toMatch(/10 above.*10 below/);
    fireEvent.click(screen.getByRole("button", { name: /one way round/i }));
    expect(summary()).toMatch(/20 above.*0 below/);
    // A band is a labelled region too, so query the control by role.
    fireEvent.change(screen.getByRole("slider", { name: /^demonstrations/i }), { target: { value: "8" } });
    expect(summary()).toMatch(/8 above/);
    fireEvent.change(screen.getByLabelText(/obstacle offset/i), { target: { value: "0.1" } });
    fireEvent.change(screen.getByLabelText(/^seed$/i), { target: { value: "4" } });
    expect(hookState.start).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    const run = screen.getByTestId("slot-3");
    fireEvent.click(within(run).getByTestId("train-button"));
    expect(hookState.start).toHaveBeenCalledWith(
      expect.objectContaining({ algorithm: "behaviour-cloning", mix: "one", demos: 8, obstacle: 0.1, seed: 4 }),
      null,
    );
    expect(within(run).getByRole("button", { name: /stop/i })).toBeDisabled();
  });

  it("puts the failure beside its control once both mixes are trained at one seed", () => {
    const both = record("both");
    const one = record("one");
    Object.assign(hookState, { history: [both, one], render: one.result.render, result: one.result });
    renderPage();
    expect(screen.getByTestId("cloning-verdict")).toHaveAttribute("data-outcome", "reached");
    expect(screen.getByTestId("verdict-both")).toHaveAttribute("data-outcome", "collided");
    expect(screen.getByTestId("verdict-one")).toHaveAttribute("data-outcome", "reached");
  }, 20_000);
});
