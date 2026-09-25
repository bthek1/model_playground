import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

// ECharts needs a layout engine happy-dom does not have; the charts themselves
// are asserted in the E2E suite.
vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const { Route } = await import("@/routes/time-series-forecasting");
const Page = Route?.options?.component as React.ComponentType | undefined;

/** Monthly, with a clear yearly period, long enough to backtest. */
function monthly(years = 8): string {
  const rows: string[] = [];
  for (let y = 0; y < years; y++) {
    for (let m = 0; m < 12; m++) {
      // A deterministic wobble on top of the trend and the season. Without it
      // the series is *exactly* periodic, seasonal naive is out by the trend
      // every step, and MASE is 1.000 in every window — which is a lovely
      // demonstration and makes "the numbers moved" untestable.
      const wobble = ((y * 7 + m * 5) % 11) - 5;
      const value = 100 + y * 12 + [0, 4, 9, 14, 20, 28, 34, 30, 22, 14, 7, 2][m] + wobble;
      rows.push(`20${String(10 + y).padStart(2, "0")}-${String(m + 1).padStart(2, "0")}-01,${value}`);
    }
  }
  return rows.join("\n");
}

function renderPage() {
  if (!Page) throw new Error("Time series forecasting route component not found");
  render(<Page />);
}

function paste(text: string) {
  fireEvent.change(screen.getByLabelText(/paste a column/i), {
    target: { value: text },
  });
}

describe("TimeSeriesForecastingPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders three bands, and says why the fourth is missing", () => {
    // Three, not four. The absence is the category's point, so the page
    // explains it where the band would have been — an unexplained gap reads as
    // an oversight. model-page-pattern.md §7 records the bend.
    renderPage();
    expect(screen.getByTestId("slot-1")).toBeInTheDocument();
    expect(screen.getByTestId("slot-2")).toBeInTheDocument();
    expect(screen.getByTestId("slot-3")).toBeInTheDocument();
    expect(screen.queryByTestId("slot-4")).toBeNull();
    expect(screen.getByTestId("no-fit-band")).toHaveTextContent(/nothing to download/i);
    expect(screen.getByTestId("no-fit-band")).toHaveTextContent(/no ONNX weights/i);
  });

  it("states it has no learned model before any forecast exists", () => {
    // The third page in the repo whose honesty note is a correctness
    // requirement, after /video-classification's frame-level disclaimer and
    // /question-answering's cannot-abstain note. In OUTPUT's *empty* state, so
    // it is read before there is a result to believe.
    renderPage();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    expect(screen.getByTestId("no-model-note")).toHaveTextContent(
      /no learned model on this page/i,
    );
  });

  it("constructs no worker at any point", () => {
    // "No worker" is the design, and a later refactor could quietly add one.
    const spy = vi.fn();
    const original = (globalThis as { Worker?: unknown }).Worker;
    (globalThis as { Worker?: unknown }).Worker = spy;
    try {
      renderPage();
      paste(monthly());
      fireEvent.click(screen.getByRole("button", { name: /^drift$/i }));
      fireEvent.change(screen.getByLabelText(/season length/i), { target: { value: "6" } });
      expect(spy).not.toHaveBeenCalled();
    } finally {
      (globalThis as { Worker?: unknown }).Worker = original;
    }
  });

  it("makes no network request at all", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    renderPage();
    paste(monthly());
    await waitFor(() => expect(screen.getByTestId("metric-table")).toBeInTheDocument());
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it("derives the forecast, the metrics and the backtest from a pasted series", () => {
    renderPage();
    paste(monthly());
    expect(screen.getByTestId("series-summary")).toHaveTextContent(/96 points/);
    expect(screen.getByTestId("series-chart")).toBeInTheDocument();
    expect(screen.getByTestId("metric-table")).toBeInTheDocument();
    expect(screen.getByTestId("backtest-strip")).toBeInTheDocument();
    expect(screen.queryByTestId("output-empty")).toBeNull();
  });

  it("renders the single-split number next to the window spread", () => {
    // The whole demonstration is the gap between them. A page that showed only
    // the spread would be correct and would not make the point.
    renderPage();
    paste(monthly());
    const line = screen.getByTestId("backtest-spread");
    expect(line).toHaveTextContent(/one split/);
    expect(line).toHaveTextContent(/windows/);
    expect(line).toHaveTextContent(/median/);
  });

  it("scores all four methods on the same held-out window", () => {
    renderPage();
    paste(monthly());
    const table = within(screen.getByTestId("metric-table"));
    for (const name of [/naive/i, /seasonal naive/i, /drift/i, /historical mean/i]) {
      expect(table.getAllByRole("row").some((r) => name.test(r.textContent ?? ""))).toBe(true);
    }
    expect(table.getByText(/MASE is the column to read/i)).toBeInTheDocument();
  });

  it("re-derives when the season changes, with no asynchronous work", () => {
    // The season scales MASE and moves every seasonal-naive number, so a page
    // that did not re-derive would show stale figures beside a changed control.
    renderPage();
    paste(monthly());
    const before = screen.getByTestId("metric-table").textContent;
    fireEvent.change(screen.getByLabelText(/season length/i), { target: { value: "5" } });
    expect(screen.getByTestId("metric-table").textContent).not.toBe(before);
  });

  it("re-derives when the horizon and the window settings change", () => {
    renderPage();
    paste(monthly());
    const before = screen.getByTestId("backtest-spread").textContent;
    fireEvent.change(screen.getByLabelText(/^Windows/), { target: { value: "4" } });
    expect(screen.getByTestId("backtest-spread").textContent).not.toBe(before);

    const after = screen.getByTestId("backtest-spread").textContent;
    fireEvent.change(screen.getByLabelText(/horizon/i), { target: { value: "3" } });
    expect(screen.getByTestId("backtest-spread").textContent).not.toBe(after);
  });

  it("switches the backtest method without touching the metric table's four rows", () => {
    renderPage();
    paste(monthly());
    const rowsBefore = within(screen.getByTestId("metric-table")).getAllByRole("row").length;
    fireEvent.click(screen.getByRole("button", { name: /^drift$/i }));
    expect(within(screen.getByTestId("metric-table")).getAllByRole("row")).toHaveLength(
      rowsBefore,
    );
    // …and the backtest is now drift's.
    expect(screen.getByText(/Rolling-origin backtest — Drift/)).toBeInTheDocument();
  });

  it("offers a sliding window length only when sliding is chosen", () => {
    renderPage();
    paste(monthly());
    expect(screen.queryByLabelText(/sliding length/i)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^sliding$/i }));
    expect(screen.getByLabelText(/sliding length/i)).toBeInTheDocument();
  });

  it("reports a gap rather than filling it", () => {
    renderPage();
    paste("2024-01-01,1\n2024-01-02,2\n2024-01-05,3\n2024-01-06,4\n2024-01-07,5\n");
    expect(screen.getByTestId("series-gaps")).toHaveTextContent(/reported, not filled/i);
  });

  it("puts a parse error in the band that produced it, and keeps the page up", () => {
    renderPage();
    paste("1\n2\noops\n");
    expect(within(screen.getByTestId("slot-2")).getByTestId("error-note")).toHaveTextContent(
      /Line 3/,
    );
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("says the season is set by the user, never detected", () => {
    renderPage();
    paste(monthly());
    expect(screen.getByText(/Set by you, never detected/i)).toBeInTheDocument();
  });
});
