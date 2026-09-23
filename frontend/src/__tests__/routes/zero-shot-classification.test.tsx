import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseZeroShotTextResult } from "@/hooks/useZeroShotText";

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

const SCORES = [
  { label: "billing", score: 0.91 },
  { label: "outage", score: 0.06 },
  { label: "feature request", score: 0.03 },
];

const mockRun = vi.fn().mockResolvedValue(SCORES);
const base: UseZeroShotTextResult = {
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  running: false,
  error: null,
  result: null,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};

let state: UseZeroShotTextResult = { ...base };
/** The id the hook was last keyed on — selecting a model must re-key it. */
let lastModel: string | null = null;
const useZeroShotText = vi.fn((model: string) => {
  lastModel = model;
  return state;
});

vi.mock("@/hooks/useZeroShotText", () => ({
  useZeroShotText: (...args: unknown[]) =>
    useZeroShotText(...(args as [string])),
}));

const { Route } = await import("@/routes/zero-shot-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Zero-shot classification route not found");
  render(<Page />);
}

const DEBERTA = "Xenova/nli-deberta-v3-xsmall";

const ready = (extra: Partial<UseZeroShotTextResult> = {}) => ({
  ...base,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

const textBox = () => screen.getByLabelText(/text to classify/i);
const labelBox = () => screen.getByLabelText(/^labels/i);
const templateBox = () => screen.getByLabelText(/hypothesis template/i);
const trigger = () => screen.getByRole("button", { name: /^classify$/i });

describe("ZeroShotClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    lastModel = null;
    localStorage.clear();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /zero-shot classification/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /deberta v3 xsmall/i, pressed: true }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /bart-large mnli/i }).length,
    ).toBeGreaterThan(0);
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves: the hook is handed no auto-load argument at all (its default
    // is `idle`), *and* nothing called load() behind the user's back.
    expect(useZeroShotText).toHaveBeenCalledWith(DEBERTA);
    expect(base.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^load model$/i }));
    expect(base.load).toHaveBeenCalledOnce();
  });

  it("stays idle after a refresh even with a stale auto-resume key", () => {
    // The key a much older build wrote. `store/models.ts` declares `partialize`
    // so it cannot revive auto-resume, and arriving at a page — cached weights
    // or not — is never a request for a download.
    localStorage.setItem(
      "model-prefs",
      JSON.stringify({ state: { selected: {}, autoResume: true }, version: 0 }),
    );
    renderPage();
    expect(base.load).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    for (const slot of ["slot-1", "slot-2", "slot-3", "slot-4"]) {
      expect(screen.getByTestId(slot)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("gates the trigger on ready, but never the text, labels or template", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(textBox()).toBeEnabled();
    expect(labelBox()).toBeEnabled();
    expect(templateBox()).toBeEnabled();
  });

  // The page's cost model, and the reason it is on screen at all.
  it("derives the pass count from the label list, spending nothing", () => {
    state = ready();
    renderPage();

    expect(screen.getByTestId("pass-count")).toHaveTextContent(
      /3 labels = 3 forward passes/i,
    );

    fireEvent.change(labelBox(), {
      target: { value: "billing\noutage\nfeature request\npraise" },
    });
    expect(screen.getByTestId("pass-count")).toHaveTextContent(
      /4 labels = 4 forward passes/i,
    );
    // Editing labels is a derivation. It must not have asked the model anything.
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("says one pass for one label, rather than pluralising blindly", () => {
    state = ready();
    renderPage();
    fireEvent.change(labelBox(), { target: { value: "urgent" } });
    expect(screen.getByTestId("pass-count")).toHaveTextContent(
      /1 label = 1 forward pass/i,
    );
  });

  // The single most valuable assertion on any task page: an input that
  // silently starts an inference looks exactly like a working page.
  it("runs nothing while the user edits any of the three inputs", () => {
    state = ready();
    renderPage();

    fireEvent.change(textBox(), { target: { value: "the server is down" } });
    fireEvent.change(labelBox(), { target: { value: "outage\nbilling" } });
    fireEvent.change(templateBox(), { target: { value: "It is about {}." } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it fills both boxes", () => {
    state = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /a news sentence/i }));

    expect(textBox()).toHaveValue(
      "The central bank held rates steady, citing softer wage growth in the services sector.",
    );
    expect(labelBox()).toHaveValue("economics\nsport\ntechnology\nhealth");
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  // Multi-label changes the arithmetic and cannot re-derive, so it is the one
  // control on this page that legitimately costs a second inference — and it
  // must still cost it on GENERATE, not on the flip.
  it("runs nothing when multi-label is flipped, and sends it on the next press", async () => {
    state = ready();
    renderPage();

    fireEvent.click(screen.getByTestId("multi-label-toggle"));
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toMatchObject({ multiLabel: true });
  });

  it("classifies when — and only when — the trigger is pressed", async () => {
    state = ready();
    renderPage();

    fireEvent.change(textBox(), { target: { value: "charged twice" } });
    fireEvent.change(labelBox(), { target: { value: "billing, outage" } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith({
      text: "charged twice",
      labels: ["billing", "outage"],
      multiLabel: false,
      hypothesisTemplate: "This example is {}.",
    });
  });

  it("sends the template the user can see, never the pipeline's own default", () => {
    // The /zero-shot-image-classification finding transplanted: the pipeline
    // applies "This example is {}." unless told otherwise, so a page that
    // templates silently is comparing a prompt the user cannot read.
    state = ready();
    renderPage();
    expect(screen.getByTestId("composed-hypothesis")).toHaveTextContent(
      "This example is billing.",
    );

    fireEvent.change(templateBox(), { target: { value: "{}" } });
    expect(screen.getByTestId("composed-hypothesis")).toHaveTextContent(
      /“billing”/,
    );

    fireEvent.click(trigger());
    return waitFor(() =>
      expect(mockRun.mock.calls[0][0]).toMatchObject({
        hypothesisTemplate: "{}",
      }),
    );
  });

  it("refuses a template with no placeholder rather than scoring nonsense", () => {
    // Without `{}` every label composes to the *same* hypothesis, so every
    // label gets the same logits and the ranking is arbitrary — with nothing
    // failing and a perfectly ordinary bar chart on screen.
    state = ready();
    renderPage();

    fireEvent.change(templateBox(), { target: { value: "This is about money." } });

    expect(screen.getByTestId("template-problem")).toBeInTheDocument();
    expect(screen.queryByTestId("composed-hypothesis")).not.toBeInTheDocument();
    expect(trigger()).toBeDisabled();
  });

  it("refuses an empty label list and an empty text", () => {
    state = ready();
    renderPage();

    fireEvent.change(labelBox(), { target: { value: "  \n , " } });
    expect(screen.getByTestId("pass-count")).toHaveTextContent(/no labels yet/i);
    expect(trigger()).toBeDisabled();

    fireEvent.change(labelBox(), { target: { value: "billing" } });
    fireEvent.change(textBox(), { target: { value: "   " } });
    expect(trigger()).toBeDisabled();
  });

  it("drops a label from the chip row without touching the model", () => {
    state = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /remove outage/i }));

    expect(labelBox()).toHaveValue("billing\nfeature request");
    expect(screen.getByTestId("pass-count")).toHaveTextContent(/2 labels/i);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("labels the result with the label set and template captured inside the run", async () => {
    state = ready();
    renderPage();

    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("ran-labels")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("ran-labels")).toHaveTextContent(
      "billing, outage, feature request",
    );
    expect(screen.getByTestId("ran-template")).toHaveTextContent(
      "This example is {}.",
    );

    // Editing the inputs afterwards must not relabel a result already on
    // screen: every one of these changes what the scores mean.
    fireEvent.change(labelBox(), { target: { value: "sport\nweather" } });
    fireEvent.change(templateBox(), { target: { value: "{}" } });
    fireEvent.change(textBox(), { target: { value: "something else" } });

    expect(screen.getByTestId("ran-labels")).toHaveTextContent(
      "billing, outage, feature request",
    );
    expect(screen.getByTestId("ran-template")).toHaveTextContent(
      "This example is {}.",
    );
    expect(screen.getByTestId("ran-text")).toHaveTextContent(/charged twice/i);
  });

  it("renders every label with its score", async () => {
    state = ready();
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("score-list")).toBeInTheDocument(),
    );
    // Scoped to the list: every one of these labels is also a chip in INPUT,
    // so an unscoped query matches the input the run was given rather than the
    // answer it produced.
    const rows = screen.getByTestId("score-list").querySelectorAll("li");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent("billing");
    expect(rows[0]).toHaveTextContent("0.91");
    expect(rows[2]).toHaveTextContent("feature request");
  });

  it("says which normalisation produced the scores it is showing", async () => {
    state = ready();
    renderPage();
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("scoring-note")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("scoring-note")).toHaveTextContent(
      /single-label.*sum to 1/i,
    );
  });

  it("gates the heaviest model behind a second opt-in, and only that one", () => {
    renderPage();
    expect(screen.queryByTestId("heavy-model-notice")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /bart-large mnli/i }),
    );

    // 816 MB on WebGPU — the roadmap's "411 MB" was its q8 size, which is not
    // what `loadOpts()` asks for on a machine with an adapter.
    const notice = screen.getByTestId("heavy-model-notice");
    // 778 MiB — `formatBytes` divides by 1024 and writes "MB", so the decimal
    // 816 MB the roadmap quotes is the same download under a different unit.
    expect(notice).toHaveTextContent(/778 MB/);
    // The selection reached the hook, so LOAD would fetch BART rather than the
    // default — and stating the cost is still not spending it.
    expect(lastModel).toBe("Xenova/bart-large-mnli");
    expect(base.load).not.toHaveBeenCalled();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = { ...base, status: "error", idle: false, error: "404 model not found" };
    renderPage();

    const note = screen.getByText(/404 model not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    state = ready({ error: "Input is too long" });
    renderPage();

    const note = screen.getByText(/input is too long/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
