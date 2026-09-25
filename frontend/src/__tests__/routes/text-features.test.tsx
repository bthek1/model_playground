import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseTextEmbedResult } from "@/hooks/useTextEmbed";
import { EMBED_MODELS } from "@/text/catalogue";

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
const BGE = EMBED_MODELS[1];

/** A 384-d vector whose tail carries most of its length, so truncating bites. */
function vector(dim = 384): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = Math.sin(i) * (i > dim / 2 ? 4 : 1);
  return v;
}

const mockRun = vi.fn(async () => [vector()]);
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

const { Route } = await import("@/routes/text-features");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text features route component not found");
  render(<Page />);
}

const box = () => screen.getByLabelText(/text to embed/i);
const trigger = () => screen.getByRole("button", { name: /^embed$/i });

function ready(extra: Partial<UseTextEmbedResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("TextFeaturesPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    states = new Map();
    state = { ...base };
    localStorage.clear();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /text features/i }),
    ).toBeInTheDocument();
    for (const m of EMBED_MODELS) {
      expect(
        screen.getAllByRole("button", { name: new RegExp(m.label, "i") }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves: no auto-load argument (the default is `idle`), and nothing
    // called load() behind the user's back.
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

  // The two catalogue facts nothing downstream can discover, on screen before
  // a byte is downloaded.
  it("states the width and the pooling before anything is loaded", () => {
    renderPage();
    expect(screen.getByTestId("embed-dim")).toHaveTextContent("384");
    expect(screen.getByTestId("embed-pooling")).toHaveTextContent("mean");
  });

  it("gates the trigger on ready, but never the input surface", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(box()).toBeEnabled();
    expect(screen.getByText(/load a model to embed text/i)).toBeInTheDocument();
  });

  it("runs nothing while the user types, even with a model ready", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "a new sentence" } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills the box", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /a recipe line/i }));

    expect(box()).toHaveValue(
      "Fold the melted butter into the flour until no dry patches remain.",
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("embeds when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "hello there" } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith("hello there");
  });

  it("draws the vector, its width and its length", async () => {
    ready();
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("embedding")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("vector-strip-dim")).toHaveTextContent("384-d");
    expect(screen.getByTestId("vector-strip-norm")).toHaveTextContent("1.000");
  });

  // The rule this page shares with /vad's threshold: arithmetic over a result
  // already in hand never re-runs the model.
  it("truncates without running anything, and reports what the cut cost", async () => {
    ready();
    renderPage();
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("embedding")).toBeInTheDocument(),
    );
    expect(mockRun).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTestId("truncate-96"));

    expect(mockRun).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("vector-strip-dim")).toHaveTextContent("96-d");
    // Renormalised at every width — that is what stops the cut scaling every
    // downstream cosine by an arbitrary factor.
    expect(screen.getByTestId("vector-strip-norm")).toHaveTextContent("1.000");
    // And the fraction actually retained is reported rather than implied.
    expect(screen.getByTestId("truncate-kept")).toHaveTextContent(
      /first 96 of 384 dimensions hold/i,
    );
  });

  it("does not relabel a finished vector when the box is edited", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "the first sentence" } });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("embedding")).toBeInTheDocument(),
    );

    fireEvent.change(box(), { target: { value: "something else entirely" } });
    expect(screen.getByTestId("embedding")).toHaveTextContent(
      "the first sentence",
    );
  });

  it("switching model spends nothing", () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getAllByRole("button", { name: new RegExp(BGE.label, "i") })[0],
    );

    expect(mockRun).not.toHaveBeenCalled();
    expect(base.load).not.toHaveBeenCalled();
  });

  // A prefix is an instruction glued to the user's text, so the page shows it
  // rather than sending something the user never saw — the same rule
  // /zero-shot-classification applies to its hypothesis template.
  it("names the task prefix for a checkpoint that needs one", () => {
    const nomic = EMBED_MODELS.find((m) => m.prefixes)!;
    ready({ meta: nomic, compose: (t: string) => `clustering: ${t}` });
    renderPage();

    fireEvent.click(
      screen.getAllByRole("button", { name: new RegExp(nomic.label, "i") })[0],
    );

    expect(screen.getByTestId("embed-prefix")).toHaveTextContent(
      nomic.prefixes!.symmetric,
    );
    // Selecting it is still free.
    expect(mockRun).not.toHaveBeenCalled();
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
    ready({ error: "the pooling option did not reach the model" });
    renderPage();

    const note = screen.getByText(/the pooling option did not reach the model/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
