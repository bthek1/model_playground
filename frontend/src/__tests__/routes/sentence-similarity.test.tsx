import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseTextEmbedResult } from "@/hooks/useTextEmbed";
import { EMBED_MODELS, PAIR_SAMPLES } from "@/text/catalogue";

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

const MINILM = EMBED_MODELS[0];
const DIM = 384;

/**
 * Deterministic vectors keyed by text, so a score is predictable: `near` texts
 * share their leading half, `far` ones do not.
 */
function vectorFor(text: string): Float32Array {
  const v = new Float32Array(DIM);
  const seed = text.includes("guitar") || text.includes("busker") ? 1 : 2;
  for (let i = 0; i < DIM; i++) {
    v[i] = i < DIM / 2 ? Math.sin(i * seed) : Math.cos(i * seed) * 3;
  }
  return v;
}

const mockRun = vi.fn(async (texts: string | string[]) =>
  (Array.isArray(texts) ? texts : [texts]).map(vectorFor),
);

const base: UseTextEmbedResult = {
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
  meta: MINILM,
  run: mockRun,
  cached: 0,
  clearCache: vi.fn(),
  compose: (t: string) => t,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};

// Keyed by model id, so a test can assert *which* entry the page asked for —
// the pooling and the width are per model, and picking the wrong one is this
// page's silent bug.
let states = new Map<string, UseTextEmbedResult>();
let state: UseTextEmbedResult = { ...base };
const useTextEmbed = vi.fn((id: string) => states.get(id) ?? state);
vi.mock("@/hooks/useTextEmbed", () => ({
  useTextEmbed: (...args: unknown[]) => useTextEmbed(...(args as [string])),
}));

const { Route } = await import("@/routes/sentence-similarity");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Sentence similarity route component not found");
  render(<Page />);
}

const fieldA = () => screen.getByLabelText(/sentence a/i);
const fieldB = () => screen.getByLabelText(/sentence b/i);
const trigger = () => screen.getByRole("button", { name: /^compare$/i });

function ready(extra: Partial<UseTextEmbedResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("SentenceSimilarityPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    states = new Map();
    state = { ...base };
    localStorage.clear();
  });

  it("renders the heading and the model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /sentence similarity/i }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: new RegExp(MINILM.label, "i") })
        .length,
    ).toBeGreaterThan(0);
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useTextEmbed).toHaveBeenCalledWith(MINILM.id);
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

  it("gates both triggers on ready, but never the two fields", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(fieldA()).toBeEnabled();
    expect(fieldB()).toBeEnabled();
  });

  it("runs nothing while either side is edited", () => {
    ready();
    renderPage();
    fireEvent.change(fieldA(), { target: { value: "one" } });
    fireEvent.change(fieldB(), { target: { value: "two" } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample pair is picked — it only fills both boxes", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /a negation/i }));

    const negation = PAIR_SAMPLES.find((p) => p.id === "negation")!;
    expect(fieldA()).toHaveValue(negation.a);
    expect(fieldB()).toHaveValue(negation.b);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("compares when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(fieldA(), { target: { value: "a guitar" } });
    fireEvent.change(fieldB(), { target: { value: "a busker" } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("similarity")).toBeInTheDocument(),
    );
    // Both sides in one batched call.
    expect(mockRun).toHaveBeenCalledTimes(1);
    expect(mockRun).toHaveBeenCalledWith(["a guitar", "a busker"]);
  });

  // A single score has no scale, which is why the sample set exists at all.
  it("scores every sample pair in one call and ranks them", async () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getByRole("button", {
        name: new RegExp(`score all ${PAIR_SAMPLES.length} sample pairs`, "i"),
      }),
    );

    await waitFor(() =>
      expect(screen.getByTestId("similarity")).toBeInTheDocument(),
    );
    expect(mockRun).toHaveBeenCalledTimes(1);
    expect((mockRun.mock.calls[0][0] as string[]).length).toBe(
      PAIR_SAMPLES.length * 2,
    );
    for (const p of PAIR_SAMPLES) {
      expect(screen.getByTestId(`pair-${p.id}`)).toBeInTheDocument();
    }
  });

  it("carries each pair's caveat beside its score", async () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getByRole("button", {
        name: new RegExp(`score all ${PAIR_SAMPLES.length} sample pairs`, "i"),
      }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("pair-negation")).toBeInTheDocument(),
    );
    // The negation pair's honest expectation ships with it: a high score there
    // is a limitation of embeddings, not a bug in the page.
    expect(
      within(screen.getByTestId("pair-negation")).getByText(
        /limitation of embeddings/i,
      ),
    ).toBeInTheDocument();
  });

  it("truncates without running anything, and re-scores from the vectors in hand", async () => {
    ready();
    renderPage();
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("similarity")).toBeInTheDocument(),
    );
    const before = screen.getByTestId("score-yours").textContent;
    expect(mockRun).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTestId("truncate-96"));

    expect(mockRun).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("strip-a-dim")).toHaveTextContent("96-d");
    // Renormalised on both sides, so the 96-d cosine is comparable with the
    // 384-d one rather than scaled by an arbitrary factor.
    expect(screen.getByTestId("strip-a-norm")).toHaveTextContent("1.000");
    expect(screen.getByTestId("strip-b-norm")).toHaveTextContent("1.000");
    // The score is allowed to move — that is the measurement — but it must
    // still be a number.
    expect(screen.getByTestId("score-yours").textContent).toMatch(/-?\d\.\d{3}/);
    expect(before).toMatch(/-?\d\.\d{3}/);
  });

  it("does not relabel a finished result when a field is edited", async () => {
    ready();
    renderPage();
    fireEvent.change(fieldA(), { target: { value: "the original A" } });
    fireEvent.change(fieldB(), { target: { value: "the original B" } });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("similarity")).toBeInTheDocument(),
    );

    fireEvent.change(fieldA(), { target: { value: "edited afterwards" } });
    expect(screen.getByTestId("pair-yours")).toHaveTextContent("the original A");
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = { ...base, status: "error", idle: false, error: "404 model not found" };
    renderPage();

    const note = screen.getByText(/404 model not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT", () => {
    ready({ error: "Asked for 2 embeddings, got 1" });
    renderPage();

    const note = screen.getByText(/asked for 2 embeddings, got 1/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
