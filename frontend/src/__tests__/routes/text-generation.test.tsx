import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseTextGenResult } from "@/hooks/useTextGen";
import { DECODING_PRESETS, TEXTGEN_MODELS } from "@/text/catalogue";

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

let probe: string | null = "webgpu";
vi.mock("@/model/useBackendProbe", () => ({
  useBackendProbe: () => probe,
}));

/** Model labels carry parentheses ("GPT-2 (124M)"), so they need escaping. */
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const SMOL = TEXTGEN_MODELS[0];
const GPT2 = TEXTGEN_MODELS[1];

const mockRun = vi.fn(async (prompt: string, decoding: unknown) => {
  void prompt;
  void decoding;
  return { text: "Paris, the capital city.", tokens: 5, ms: 500 };
});

const base: UseTextGenResult = {
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
  meta: SMOL,
  result: null,
  partial: null,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};

let state: UseTextGenResult = { ...base };
const useTextGen = vi.fn((id: string) => ({
  ...state,
  meta: TEXTGEN_MODELS.find((m) => m.id === id) ?? state.meta,
}));
vi.mock("@/hooks/useTextGen", () => ({
  useTextGen: (...args: unknown[]) => useTextGen(...(args as [string])),
}));

const { Route } = await import("@/routes/text-generation");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text generation route component not found");
  render(<Page />);
}

// By role: the RUN band is a `region` labelled "Prompt" too, so
// `getByLabelText` matches both the band and the field inside it.
const box = () => screen.getByRole("textbox");
const trigger = () => screen.getByRole("button", { name: /^generate$/i });

function ready(extra: Partial<UseTextGenResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("TextGenerationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    probe = "webgpu";
    localStorage.clear();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { level: 1, name: /text generation/i }),
    ).toBeInTheDocument();
    for (const m of TEXTGEN_MODELS) {
      expect(
        screen.getAllByRole("button", { name: new RegExp(escape(m.label), "i") }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useTextGen).toHaveBeenCalledWith(SMOL.id);
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

  it("gates both triggers on ready, but never the prompt or the controls", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(screen.getByTestId("compare-run")).toBeDisabled();
    expect(box()).toBeEnabled();
    // The strategy is a choice, and choices are free.
    expect(screen.getByTestId("mode-sample")).toBeEnabled();
    expect(screen.getByTestId("preset-loop")).toBeEnabled();
  });

  it("runs nothing while the user types", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "Once upon a time" } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample prompt is picked", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /invites a loop/i }));

    expect(box()).toHaveValue(
      "Here is a list of things to remember when travelling:",
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  // The page's defining rule: none of these can re-derive from a finished
  // answer, so they are INPUT and they spend — but only on GENERATE.
  it("changing any decoding control runs nothing, and the page says the next run is real", () => {
    ready();
    renderPage();

    fireEvent.click(screen.getByTestId("mode-sample"));
    fireEvent.change(screen.getByLabelText(/temperature/i), {
      target: { value: "1.5" },
    });
    fireEvent.change(screen.getByLabelText(/repetition penalty/i), {
      target: { value: "1.4" },
    });
    fireEvent.click(screen.getByTestId("preset-loop"));

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("spend-note")).toHaveTextContent(
      /the next Generate is a real inference/i,
    );
    expect(screen.getByTestId("spend-note")).toHaveTextContent(/two/i);
  });

  it("greys out the sampling knobs under greedy decoding", () => {
    ready();
    renderPage();
    // Greedy is the default preset.
    expect(screen.getByLabelText(/temperature/i)).toBeDisabled();
    expect(screen.getByLabelText(/top_p/i)).toBeDisabled();
    // The repetition penalty applies to both, so it stays live.
    expect(screen.getByLabelText(/repetition penalty/i)).toBeEnabled();

    fireEvent.click(screen.getByTestId("mode-sample"));
    expect(screen.getByLabelText(/temperature/i)).toBeEnabled();
  });

  it("generates when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The capital of France is" } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toBe("The capital of France is");
    expect(mockRun.mock.calls[0][1]).toEqual(DECODING_PRESETS[0].decoding);
  });

  it("sends the edited settings on the next run", async () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByTestId("mode-sample"));
    fireEvent.change(screen.getByLabelText(/temperature/i), {
      target: { value: "1.5" },
    });
    fireEvent.click(trigger());

    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][1]).toMatchObject({
      doSample: true,
      temperature: 1.5,
    });
  });

  it("renders the stream while a run is in flight, and not the finished answer", () => {
    ready({ running: true, partial: { text: "Paris, the", tokens: 3 } });
    renderPage();

    expect(screen.getByTestId("stream")).toHaveTextContent("Paris, the");
    expect(screen.getByTestId("stream")).toHaveTextContent("3 tokens");
    // Never both: a half-finished answer beside a finished one is the thing
    // `partial` being cleared on result exists to prevent.
    expect(screen.queryByTestId("generated-text")).not.toBeInTheDocument();
    expect(screen.queryByTestId("output-running")).not.toBeInTheDocument();
  });

  it("labels the answer with the settings and prompt captured at run time", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The original prompt" } });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("generated-text")).toBeInTheDocument(),
    );

    expect(screen.getByTestId("ran-settings")).toHaveTextContent(/greedy/);
    expect(screen.getByTestId("ran-prompt")).toHaveTextContent(
      "The original prompt",
    );

    // Moving a slider afterwards must not relabel a result already on screen.
    fireEvent.click(screen.getByTestId("mode-sample"));
    fireEvent.change(box(), { target: { value: "edited afterwards" } });
    expect(screen.getByTestId("ran-settings")).toHaveTextContent(/greedy/);
    expect(screen.getByTestId("ran-prompt")).toHaveTextContent(
      "The original prompt",
    );
  });

  // The comparison is the page: two generations of one prompt, so a loop is
  // attributable to the strategy rather than to the weights.
  it("compares with two inferences of the same prompt, the second greedy", async () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByTestId("mode-sample"));
    fireEvent.change(box(), { target: { value: "Same prompt both times" } });
    fireEvent.click(screen.getByTestId("compare-run"));

    await waitFor(() =>
      expect(screen.getByTestId("comparison")).toBeInTheDocument(),
    );
    expect(mockRun).toHaveBeenCalledTimes(2);
    expect(mockRun.mock.calls[0][0]).toBe("Same prompt both times");
    expect(mockRun.mock.calls[1][0]).toBe("Same prompt both times");
    expect(mockRun.mock.calls[0][1]).toMatchObject({ doSample: true });
    expect(mockRun.mock.calls[1][1]).toMatchObject({ doSample: false });
    expect(screen.getByTestId("comparison")).toHaveTextContent(
      /is the .*strategy.*, not the model/i,
    );
  });

  // #30's finding: "resolved to wasm" is misleading on a machine that plainly
  // has a GPU, so the missing feature is named.
  it("names shader-f16 when the adapter lacks it, rather than saying 'wasm'", () => {
    // The note is about the **machine**, not the selected row — and it has to
    // be, because the rows it explains are exactly the ones the picker has
    // disabled, which the user cannot select in order to read a message on.
    probe = "wasm";
    ready();
    renderPage();

    const note = screen.getByTestId("shader-f16-note");
    expect(note).toHaveTextContent(/shader-f16/);
    expect(note).toHaveTextContent(/first operator/i);
    expect(note).toHaveTextContent(new RegExp(escape(GPT2.label), "i"));
    // The rows themselves are withheld rather than offered.
    expect(
      screen.getAllByRole("button", {
        name: new RegExp(escape(GPT2.label), "i"),
      })[0],
    ).toBeDisabled();
  });

  it("says nothing about shader-f16 on an adapter that has it", () => {
    probe = "webgpu";
    ready();
    renderPage();
    expect(screen.queryByTestId("shader-f16-note")).not.toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = {
      ...base,
      status: "error",
      idle: false,
      error: "404 model not found",
    };
    renderPage();

    const note = screen.getByText(/404 model not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT", () => {
    ready({ error: "Program Gather requires f16" });
    renderPage();

    const note = screen.getByText(/program gather requires f16/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
