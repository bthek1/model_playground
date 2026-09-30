import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseTextClassifierResult } from "@/hooks/useTextClassifier";
import {
  describeTaskPageContract,
  idleTask,
  readyTask,
  routeComponent,
} from "@/test/taskPage";

const POSITIVE = [
  { label: "POSITIVE", score: 0.97 },
  { label: "NEGATIVE", score: 0.03 },
];

const mockRun = vi.fn().mockResolvedValue(POSITIVE);
const base = idleTask<UseTextClassifierResult>({
  result: null,
  run: mockRun,
});

// The route calls the hook twice — once for the model it classifies with, once
// for the optional comparison. States are keyed by model id so a test can put
// one of them `ready` without the other.
let states = new Map<string, UseTextClassifierResult>();
let fallback: UseTextClassifierResult = { ...base };
const useTextClassifier = vi.fn(
  (id: string) => states.get(id) ?? fallback,
);

vi.mock("@/hooks/useTextClassifier", () => ({
  useTextClassifier: (...args: unknown[]) =>
    useTextClassifier(...(args as [string])),
}));

const Page = routeComponent(await import("@/routes/text-classification"));

function renderPage() {
  render(<Page />);
}

const DISTILBERT = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
const FINBERT = "Xenova/finbert";

const ready = (extra: Partial<UseTextClassifierResult> = {}) =>
  readyTask(base, extra);

/** Put the primary model in `ready`, leaving the comparison hook idle. */
function primaryReady(extra: Partial<UseTextClassifierResult> = {}) {
  states.set(DISTILBERT, ready(extra));
}

describe("TextClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    states = new Map();
    fallback = { ...base };
  });

  describeTaskPageContract({
    render: renderPage,
    hook: useTextClassifier,
    base,
    setState: (st) => {
      fallback = st;
    },
    trigger: /^classify$/i,
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /text classification/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /distilbert sst-2/i, pressed: true }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /twitter roberta/i }).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByRole("button", { name: /finbert/i }).length,
    ).toBeGreaterThan(0);
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves matter: the hook is handed no auto-load argument at all (its
    // default is `idle`), *and* nothing has called load() behind the user's
    // back.
    expect(useTextClassifier).toHaveBeenCalledWith(DISTILBERT);
    expect(base.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^load model$/i }));
    expect(base.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getByTestId("slot-1")).toBeInTheDocument();
    expect(screen.getByTestId("slot-2")).toBeInTheDocument();
    expect(screen.getByTestId("slot-3")).toBeInTheDocument();
    expect(screen.getByTestId("slot-4")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the trigger disabled until a model is ready, but not the textarea", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
    // The input surface is never gated on `ready` — writing the sentence first
    // is free and commits to nothing (model-page-pattern.md §1.2).
    expect(screen.getByLabelText(/text to classify/i)).toBeEnabled();
    expect(
      screen.getByText(/load a model to classify/i),
    ).toBeInTheDocument();
  });

  // The core of the SELECT → LOAD → INPUT → GENERATE contract, and the single
  // most valuable assertion on the page: an input that silently starts an
  // inference looks exactly like a working page.
  it("runs nothing while the user types, even with a model ready", () => {
    primaryReady();
    renderPage();

    fireEvent.change(screen.getByLabelText(/text to classify/i), {
      target: { value: "an absolute delight" },
    });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills the box", () => {
    primaryReady();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /a market report/i }));

    expect(screen.getByLabelText(/text to classify/i)).toHaveValue(
      "Shares plunged 12% after the company slashed its full-year guidance.",
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("classifies when — and only when — the trigger is pressed", async () => {
    primaryReady();
    renderPage();

    fireEvent.change(screen.getByLabelText(/text to classify/i), {
      target: { value: "an absolute delight" },
    });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith("an absolute delight");
  });

  it("refuses to run on an empty box rather than sending whitespace", () => {
    primaryReady();
    renderPage();
    fireEvent.change(screen.getByLabelText(/text to classify/i), {
      target: { value: "   " },
    });
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
  });

  it("labels the result with the text captured inside the run", async () => {
    primaryReady();
    renderPage();

    fireEvent.change(screen.getByLabelText(/text to classify/i), {
      target: { value: "the first sentence" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() =>
      expect(screen.getByTestId("ran-text")).toHaveTextContent(
        "the first sentence",
      ),
    );

    // Editing the box afterwards must not relabel a finished result.
    fireEvent.change(screen.getByLabelText(/text to classify/i), {
      target: { value: "something else entirely" },
    });
    expect(screen.getByTestId("ran-text")).toHaveTextContent(
      "the first sentence",
    );
  });

  it("renders every label with its score", async () => {
    primaryReady();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));

    await waitFor(() =>
      expect(screen.getByTestId("answer-primary")).toBeInTheDocument(),
    );
    expect(screen.getByText("POSITIVE")).toBeInTheDocument();
    expect(screen.getByText("0.97")).toBeInTheDocument();
    expect(screen.getByText("NEGATIVE")).toBeInTheDocument();
  });

  // The head-to-head is a second download, not a toggle over a held result.
  it("states the comparison model's cost before the load button that spends it", () => {
    primaryReady();
    renderPage();
    expect(screen.queryByTestId("compare-load")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getAllByRole("button", { name: /^finbert$/i })[0],
    );

    expect(screen.getByTestId("compare-cost")).toHaveTextContent(/MB/);
    expect(screen.getByTestId("compare-load")).toBeInTheDocument();
    // Choosing it downloads nothing — there is a second LOAD button for that.
    expect(base.load).not.toHaveBeenCalled();
  });

  it("runs both models on one press once the comparison is loaded", async () => {
    const compareRun = vi.fn().mockResolvedValue([
      { label: "neutral", score: 0.8 },
      { label: "positive", score: 0.2 },
    ]);
    primaryReady();
    states.set(FINBERT, ready({ run: compareRun }));
    renderPage();

    fireEvent.click(screen.getAllByRole("button", { name: /^finbert$/i })[0]);
    fireEvent.click(
      screen.getByRole("button", { name: /classify with both/i }),
    );

    await waitFor(() =>
      expect(screen.getByTestId("answer-compare")).toBeInTheDocument(),
    );
    expect(mockRun).toHaveBeenCalledTimes(1);
    expect(compareRun).toHaveBeenCalledTimes(1);
    // Both answers came from the same captured sentence.
    expect(mockRun.mock.calls[0][0]).toBe(compareRun.mock.calls[0][0]);
  });
});
