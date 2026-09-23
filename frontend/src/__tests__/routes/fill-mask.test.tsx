import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseFillMaskResult } from "@/hooks/useFillMask";

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

const PARIS = {
  mask: "[MASK]",
  fills: [
    [
      { label: "paris", score: 0.33 },
      { label: "lille", score: 0.09 },
    ],
  ],
};

const mockRun = vi.fn().mockResolvedValue(PARIS);
const base: UseFillMaskResult = {
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
  maskToken: "[MASK]",
};

let state: UseFillMaskResult = { ...base };
const useFillMask = vi.fn((_id?: string) => state);
vi.mock("@/hooks/useFillMask", () => ({
  useFillMask: (...args: unknown[]) => useFillMask(...(args as [string])),
}));

const { Route } = await import("@/routes/fill-mask");
const Page = Route?.options?.component as React.ComponentType | undefined;

const BERT = "Xenova/bert-base-uncased";

function renderPage() {
  if (!Page) throw new Error("Fill mask route component not found");
  render(<Page />);
}

const box = () => screen.getByLabelText(/sentence with a mask/i);
const trigger = () => screen.getByRole("button", { name: /fill the mask/i });

function ready(extra: Partial<UseFillMaskResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("FillMaskPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    localStorage.clear();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /fill mask/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /bert base \(uncased\)/i, pressed: true }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /roberta base/i }).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByRole("button", { name: /modernbert base/i }).length,
    ).toBeGreaterThan(0);
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves: the hook is handed no auto-load argument (its default is
    // `idle`), and nothing called load() behind the user's back.
    expect(useFillMask).toHaveBeenCalledWith(BERT);
    expect(base.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^load model$/i }));
    expect(base.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    for (const slot of ["slot-1", "slot-2", "slot-3", "slot-4"]) {
      expect(screen.getByTestId(slot)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("shows the mask token before anything is downloaded", () => {
    // The whole point of carrying it as catalogue data: the user can see which
    // token this checkpoint uses without paying 219 MB to find out.
    renderPage();
    expect(screen.getByTestId("mask-token")).toHaveTextContent("[MASK]");
  });

  it("gates the trigger on ready, but never the input surface", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(box()).toBeEnabled();
    expect(screen.getByTestId("insert-mask")).toBeEnabled();
    expect(screen.getByText(/load a model to fill the mask/i)).toBeInTheDocument();
  });

  it("runs nothing while the user types, even with a model ready", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The sky is [MASK]." } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills the box", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /a date stamp/i }));

    expect(box()).toHaveValue("I looked up the address on [MASK].");
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("inserts the mask without running anything", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The sky is" } });
    fireEvent.click(screen.getByTestId("insert-mask"));

    expect(box()).toHaveValue("The sky is [MASK]");
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("fills when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The sky is [MASK]." } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith("The sky is [MASK].");
  });

  // The page's own refusal, and the reason it exists: the pipeline throws on
  // zero masks and *silently drops* every mask after the first.
  it("refuses to run with no mask, and says why on the trigger", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The sky is blue." } });

    expect(trigger()).toBeDisabled();
    expect(screen.getByTestId("mask-note")).toHaveTextContent(
      /no \[MASK\] in the sentence/i,
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("refuses to run with two masks, naming what the pipeline would do", () => {
    ready();
    renderPage();
    fireEvent.change(box(), {
      target: { value: "The [MASK] of France is [MASK]." },
    });

    expect(trigger()).toBeDisabled();
    expect(screen.getByTestId("mask-note")).toHaveTextContent(
      /only fills the first/i,
    );
  });

  it("clears the refusal once exactly one mask is present", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "no mask" } });
    expect(screen.getByTestId("mask-note")).toBeInTheDocument();

    fireEvent.change(box(), { target: { value: "one [MASK] here" } });
    expect(screen.queryByTestId("mask-note")).not.toBeInTheDocument();
    expect(trigger()).toBeEnabled();
  });

  // The model-switch rule, decided in favour of rewriting rather than
  // prompting — and stated on screen either way, never silent.
  it("rewrites the mask in held text when the model changes, and says so", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The capital is [MASK]." } });

    fireEvent.click(screen.getAllByRole("button", { name: /roberta base/i })[0]);

    expect(box()).toHaveValue("The capital is <mask>.");
    expect(screen.getByTestId("mask-rewritten")).toHaveTextContent(/\[MASK\]/);
    expect(screen.getByTestId("mask-token")).toHaveTextContent("<mask>");
    // A model change is SELECT, and SELECT spends nothing.
    expect(mockRun).not.toHaveBeenCalled();
    expect(base.load).not.toHaveBeenCalled();
  });

  it("says nothing about a rewrite when there was no mask to rewrite", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "no token in here" } });
    fireEvent.click(screen.getAllByRole("button", { name: /roberta base/i })[0]);

    expect(screen.queryByTestId("mask-rewritten")).not.toBeInTheDocument();
  });

  it("splices the filling into the user's own string, not the decode", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), {
      target: { value: "The CAPITAL of France is [MASK]." },
    });
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("filled")).toBeInTheDocument(),
    );
    // The user's casing survives — an uncased model's own `sequence` would have
    // come back as "the capital of france is paris."
    expect(screen.getByTestId("span-overlay")).toHaveTextContent(
      "The CAPITAL of France is paris.",
    );
    expect(screen.getByTestId("span-mark")).toHaveTextContent("paris");
  });

  it("ranks every candidate, not only the top one", async () => {
    ready();
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("fill-scores")).toBeInTheDocument(),
    );
    // Scoped to the list: "paris" is also in the spliced sentence above it,
    // which is the point — the same word in two places, not one rendered twice.
    const scores = within(screen.getByTestId("fill-scores"));
    expect(scores.getByText("paris")).toBeInTheDocument();
    expect(scores.getByText("lille")).toBeInTheDocument();
    expect(scores.getByText("0.33")).toBeInTheDocument();
  });

  it("does not relabel a finished result when the box is edited", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The sky is [MASK]." } });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("filled")).toBeInTheDocument(),
    );

    fireEvent.change(box(), { target: { value: "something else [MASK]" } });
    expect(screen.getByTestId("span-overlay")).toHaveTextContent(
      "The sky is paris.",
    );
  });

  it("says so when the tokenizer's mask differs from the declared one", async () => {
    // The run is still correct — the engine defers to the tokenizer — but the
    // SELECT slot is showing a token that was not used, so the page says which.
    ready({
      run: vi.fn().mockResolvedValue({
        mask: "<mask>",
        fills: [[{ label: "paris", score: 0.4 }]],
      }),
    });
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("mask-drift")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("mask-drift")).toHaveTextContent("<mask>");
  });

  it("runs the bias probes as one batched call, on their own press", async () => {
    const probeRun = vi.fn().mockResolvedValue({
      mask: "[MASK]",
      fills: [
        [{ label: "lawyer", score: 0.1 }],
        [{ label: "nurse", score: 0.2 }],
        [{ label: "his", score: 0.5 }],
        [{ label: "her", score: 0.4 }],
        [{ label: "strong", score: 0.2 }],
        [{ label: "frail", score: 0.2 }],
      ],
    });
    ready({ run: probeRun });
    renderPage();

    expect(probeRun).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /run the bias probes/i }));

    await waitFor(() => expect(screen.getByTestId("probes")).toBeInTheDocument());
    // One call, six prompts — each carrying this model's own token.
    expect(probeRun).toHaveBeenCalledTimes(1);
    const prompts = probeRun.mock.calls[0][0] as string[];
    expect(prompts).toHaveLength(6);
    for (const p of prompts) expect(p).toContain("[MASK]");

    expect(screen.getByTestId("probe-occupation-man")).toHaveTextContent("lawyer");
    expect(screen.getByTestId("probe-occupation-woman")).toHaveTextContent("nurse");
    // The framing travels with the result, as §3.9 requires.
    expect(
      screen.getByText(/evidence about the training data/i),
    ).toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = { ...base, status: "error", idle: false, error: "404 model not found" };
    renderPage();

    const note = screen.getByText(/404 model not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    ready({ error: "Mask token ([MASK]) not found in text." });
    renderPage();

    // The whole sentence: "mask token" alone also matches the SELECT slot's
    // line naming this checkpoint's token.
    const note = screen.getByText("Mask token ([MASK]) not found in text.");
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
