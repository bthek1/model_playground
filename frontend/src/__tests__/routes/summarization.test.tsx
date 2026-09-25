import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseEntailmentResult } from "@/hooks/useEntailment";
import type { UseSummarizeResult } from "@/hooks/useSummarize";
import { ARTICLE_SAMPLES, SUMMARIZER_MODELS } from "@/text/catalogue";

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

vi.mock("@/model/useBackendProbe", () => ({
  useBackendProbe: () => "webgpu",
}));

const T5 = SUMMARIZER_MODELS[0];
const DISTILBART = SUMMARIZER_MODELS[1];

const SUMMARY = "The agency launched a rocket. Four satellites reached orbit.";

const mockRun = vi.fn(async (request: unknown) => {
  void request;
  return SUMMARY;
});
const base: UseSummarizeResult = {
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
  meta: T5,
  result: null,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};

const entailRun = vi.fn(async (_p: string, sentences: readonly string[]) =>
  sentences.map((sentence, i) => ({ sentence, score: i === 0 ? 0.9 : 0.2 })),
);
let entail: UseEntailmentResult = {
  ...base,
  modelId: "Xenova/mobilebert-uncased-mnli",
  result: null,
  run: entailRun,
} as unknown as UseEntailmentResult;

// Keyed by model id, so a test can assert *which* entry the page asked
// for — picking the wrong one is the bug these pages can silently have.
let states = new Map<string, UseSummarizeResult>();
let state: UseSummarizeResult = { ...base };
const useSummarize = vi.fn((id: string) => states.get(id) ?? state);
vi.mock("@/hooks/useSummarize", () => ({
  useSummarize: (...args: unknown[]) => useSummarize(...(args as [string])),
}));
vi.mock("@/hooks/useEntailment", () => ({
  useEntailment: () => entail,
}));

const { Route } = await import("@/routes/summarization");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Summarization route component not found");
  render(<Page />);
}

const box = () => screen.getByRole("textbox");
const trigger = () => screen.getByRole("button", { name: /^summarize$/i });

function ready(extra: Partial<UseSummarizeResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

function entailReady() {
  entail = {
    ...entail,
    status: "ready",
    idle: false,
    ready: true,
    run: entailRun,
  } as UseEntailmentResult;
}

describe("SummarizationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    states = new Map();
    state = { ...base };
    entail = {
      ...base,
      modelId: "Xenova/mobilebert-uncased-mnli",
      result: null,
      run: entailRun,
      load: vi.fn(),
    } as unknown as UseEntailmentResult;
    localStorage.clear();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { level: 1, name: /summarization/i }),
    ).toBeInTheDocument();
    for (const m of SUMMARIZER_MODELS) {
      expect(
        screen.getAllByRole("button", { name: new RegExp(m.label, "i") }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useSummarize).toHaveBeenCalledWith(T5.id);
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

  // Phase 1's requirement, and the page's best idea: the baseline needs no
  // model, so it *is* the empty state — on screen before anything downloads,
  // already saying what the model will be measured against.
  it("shows the lead-3 baseline from idle, with no model at all", () => {
    renderPage();
    const preview = screen.getByTestId("baseline-preview");
    expect(preview).toHaveTextContent(
      /The European Space Agency confirmed on Tuesday/,
    );
    // Three sentences of the sample, not the whole thing.
    expect(preview).not.toHaveTextContent(/Copernicus programme/);
    expect(base.load).not.toHaveBeenCalled();
  });

  it("re-derives the baseline as the article is edited, running nothing", () => {
    renderPage();
    fireEvent.change(box(), {
      target: { value: "One. Two. Three. Four should not appear." },
    });

    const preview = screen.getByTestId("baseline-preview");
    expect(preview).toHaveTextContent("One. Two. Three.");
    expect(preview).not.toHaveTextContent("Four should not appear");
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("gates the trigger on ready, but never the textarea", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(box()).toBeEnabled();
    expect(
      screen.getByText(/the lead-3 baseline already works/i),
    ).toBeInTheDocument();
  });

  it("runs nothing while the user types", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "A new article. Second." } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills the box", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /a name and a number/i }));

    expect(box()).toHaveValue(
      ARTICLE_SAMPLES.find((s) => s.id === "fabricated")!.text,
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  // The page's own statement about its length controls, asserted so it stays
  // true: they change the generation, so they cannot re-derive.
  it("changing the length controls runs nothing, and says the next run is real", () => {
    ready();
    renderPage();
    fireEvent.change(screen.getByLabelText(/max new tokens/i), {
      target: { value: "200" },
    });
    fireEvent.change(screen.getByLabelText(/min length/i), {
      target: { value: "10" },
    });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("rerun-note")).toHaveTextContent(
      /real second inference/i,
    );
  });

  it("summarizes when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "First. Second. Third." } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith({
      text: "First. Second. Third.",
      maxNewTokens: 130,
      minLength: 30,
    });
  });

  it("sends the edited length parameters on the next run", async () => {
    ready();
    renderPage();
    fireEvent.change(screen.getByLabelText(/max new tokens/i), {
      target: { value: "60" },
    });
    fireEvent.click(trigger());

    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toMatchObject({ maxNewTokens: 60 });
  });

  // The summary and its baseline must come from the *same* captured article,
  // or the page puts a summary of one text beside three sentences of another.
  it("captures the article, so neither half moves when the box is edited", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), {
      target: { value: "Alpha one. Beta two. Gamma three. Delta four." },
    });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("summarized")).toBeInTheDocument(),
    );

    expect(screen.getByTestId("baseline-text")).toHaveTextContent(
      "Alpha one. Beta two. Gamma three.",
    );
    expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY);

    fireEvent.change(box(), { target: { value: "Something else entirely." } });
    expect(screen.getByTestId("baseline-text")).toHaveTextContent("Alpha one.");
    expect(screen.getByTestId("summary-text")).toHaveTextContent(SUMMARY);
    expect(screen.getByTestId("ran-article")).toHaveTextContent("Alpha one.");
  });

  it("says a GPU-only entry is GPU-only, and why", () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getAllByRole("button", {
        name: new RegExp(DISTILBART.label, "i"),
      })[0],
    );

    expect(screen.getByTestId("gpu-only-note")).toHaveTextContent(
      /cannot be quantized/i,
    );
    // Still a SELECT change: it spends nothing.
    expect(base.load).not.toHaveBeenCalled();
    expect(mockRun).not.toHaveBeenCalled();
  });

  describe("the faithfulness check", () => {
    it("is off by default and downloads nothing", () => {
      ready();
      renderPage();
      expect(screen.queryByTestId("faithful-load")).not.toBeInTheDocument();
      expect(entail.load).not.toHaveBeenCalled();
    });

    it("opting in reveals a second LOAD but still downloads nothing", () => {
      ready();
      renderPage();
      fireEvent.click(screen.getByTestId("enable-faithfulness"));

      expect(screen.getByTestId("faithful-load")).toBeInTheDocument();
      // Opting in is a statement of intent, not a command — the same rule
      // SELECT follows.
      expect(entail.load).not.toHaveBeenCalled();
    });

    it("scores each summary sentence, one pass per sentence", async () => {
      ready();
      entailReady();
      renderPage();
      fireEvent.click(screen.getByTestId("enable-faithfulness"));
      fireEvent.click(trigger());
      await waitFor(() =>
        expect(screen.getByTestId("summarized")).toBeInTheDocument(),
      );

      // Nothing scored until asked.
      expect(entailRun).not.toHaveBeenCalled();
      fireEvent.click(screen.getByTestId("check-faithfulness"));

      await waitFor(() =>
        expect(screen.getByTestId("faithfulness-scores")).toBeInTheDocument(),
      );
      // The summary's two sentences, scored against the whole article.
      expect(entailRun).toHaveBeenCalledTimes(1);
      const sentences = entailRun.mock.calls[0][1] as string[];
      expect(sentences).toHaveLength(2);
      expect(screen.getByTestId("faithfulness-scores")).toHaveTextContent("0.90");
      expect(screen.getByTestId("faithfulness-scores")).toHaveTextContent("0.20");
    });

    it("states the per-sentence reason and the out-of-distribution caveat", async () => {
      ready();
      entailReady();
      renderPage();
      fireEvent.click(screen.getByTestId("enable-faithfulness"));
      fireEvent.click(trigger());
      await waitFor(() =>
        expect(screen.getByTestId("faithfulness")).toBeInTheDocument(),
      );

      expect(screen.getByTestId("faithfulness")).toHaveTextContent(
        /an aggregate hides the one fabricated clause/i,
      );
      expect(screen.getByTestId("faithfulness")).toHaveTextContent(
        /out of\s+distribution/i,
      );
    });
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = {
      ...base,
      status: "error",
      idle: false,
      error: "Can't create a session. qdq_actions.cc:137",
    };
    renderPage();

    const note = screen.getByText(/qdq_actions/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT", () => {
    ready({ error: "Input is too long for this encoder" });
    renderPage();

    const note = screen.getByText(/too long for this encoder/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
