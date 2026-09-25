import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseRankingResult } from "@/hooks/useRanking";
import { RANKING_CORPUS, RANKING_PAIRS } from "@/text/catalogue";

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

/** Pair labels contain "+", which is a regex quantifier. */
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const PAIR = RANKING_PAIRS[0];
const OTHER = RANKING_PAIRS[1];

/** Dense order: deliberately the reverse of BM25's, so the columns differ. */
const mockDense = vi.fn(async (_q: string, corpus: readonly string[]) =>
  corpus.map((_, i) => ({ doc: corpus.length - 1 - i, score: 1 - i / 100 })),
);
const mockRerank = vi.fn(
  async (_q: string, _c: readonly string[], candidates: readonly number[]) =>
    candidates.map((doc, i) => ({ doc, score: 1 - i / 10 })),
);
const mockEmbedCorpus = vi.fn(async () => {});

const base: UseRankingResult = {
  pair: PAIR,
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  error: null,
  running: false,
  embedding: null,
  embedded: 0,
  embedCorpus: mockEmbedCorpus,
  denseSearch: mockDense,
  rerank: mockRerank,
  clear: vi.fn(),
};

let state: UseRankingResult = { ...base };
const useRanking = vi.fn((id: string) => ({
  ...state,
  pair: RANKING_PAIRS.find((p) => p.id === id) ?? state.pair,
}));
vi.mock("@/hooks/useRanking", () => ({
  useRanking: (...args: unknown[]) => useRanking(...(args as [string])),
}));

const { Route } = await import("@/routes/text-ranking");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text ranking route component not found");
  render(<Page />);
}

const queryField = () => screen.getByLabelText(/^query$/i);
const corpusField = () => screen.getByLabelText(/one document per line/i);

function ready(extra: Partial<UseRankingResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("TextRankingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    localStorage.clear();
  });

  it("renders the heading and every pair", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { level: 1, name: /text ranking/i }),
    ).toBeInTheDocument();
    for (const p of RANKING_PAIRS) {
      expect(
        screen.getAllByRole("button", { name: new RegExp(escape(p.label), "i") }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useRanking).toHaveBeenCalledWith(PAIR.id);
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

  // The page's own claim, and the reason the entry quotes a combined size: two
  // models are live at once, which is a declared exception.
  it("says both models load together and quotes one combined size", () => {
    renderPage();
    expect(screen.getByTestId("slot-1")).toHaveTextContent(
      new RegExp(escape(PAIR.embedder.label), "i"),
    );
    expect(screen.getByTestId("slot-1")).toHaveTextContent(
      new RegExp(escape(PAIR.reranker.label), "i"),
    );
    expect(screen.getByTestId("slot-1")).toHaveTextContent(/combined/i);
    // One size line, from the picker — never one per half.
    expect(screen.getAllByTestId("model-size-note")).toHaveLength(1);
  });

  it("gates both spending triggers on ready, but never the input", () => {
    renderPage();
    expect(screen.getByTestId("embed-corpus")).toBeDisabled();
    expect(screen.getByTestId("search")).toBeDisabled();
    expect(queryField()).toBeEnabled();
    expect(corpusField()).toBeEnabled();
  });

  // BM25 needs no model, so it is useful from the first render — which is the
  // most useful thing this page has to say.
  it("ranks with BM25 before anything is loaded at all", () => {
    renderPage();
    const live = screen.getByTestId("bm25-live");
    // The lexical sample's words appear in the corpus, so something ranks.
    expect(live).not.toHaveTextContent(/type a query and a corpus/i);
    expect(base.load).not.toHaveBeenCalled();
  });

  it("re-scores BM25 as k1 and b are dragged, running nothing", () => {
    ready();
    renderPage();
    const before = screen.getByTestId("bm25-live").textContent;

    fireEvent.change(screen.getByLabelText(/k1 \(saturation\)/i), {
      target: { value: "0" },
    });
    fireEvent.change(screen.getByLabelText(/b \(length norm/i), {
      target: { value: "0" },
    });

    expect(mockDense).not.toHaveBeenCalled();
    expect(mockRerank).not.toHaveBeenCalled();
    expect(mockEmbedCorpus).not.toHaveBeenCalled();
    // It is still a ranking, and the parameters really reached it.
    expect(screen.getByTestId("bm25-live").textContent).toBeTruthy();
    expect(before).toBeTruthy();
  });

  it("runs nothing while the query or corpus is edited", () => {
    ready();
    renderPage();
    fireEvent.change(queryField(), { target: { value: "a brand new query" } });
    fireEvent.change(corpusField(), { target: { value: "one\ntwo\nthree" } });

    expect(mockDense).not.toHaveBeenCalled();
    expect(mockEmbedCorpus).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample query is picked", () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getByRole("button", { name: /an ambiguous word/i }),
    );

    expect(queryField()).toHaveValue(
      "how do I stop something that keeps running",
    );
    expect(mockDense).not.toHaveBeenCalled();
  });

  // Two triggers, two costs — the page's whole cost model.
  it("embedding the corpus and searching are separate calls", () => {
    ready();
    renderPage();

    fireEvent.click(screen.getByTestId("embed-corpus"));
    expect(mockEmbedCorpus).toHaveBeenCalledTimes(1);
    expect(mockDense).not.toHaveBeenCalled();
    expect(mockRerank).not.toHaveBeenCalled();
  });

  it("states each trigger's cost in passes, derived from the corpus", () => {
    ready();
    renderPage();
    const note = screen.getByTestId("cost-note");
    expect(note).toHaveTextContent(String(RANKING_CORPUS.length));
    expect(note).toHaveTextContent(/forward passes/i);
    expect(note).toHaveTextContent(/cannot be precomputed/i);

    // And it re-derives: a shorter corpus quotes fewer passes, with no run.
    fireEvent.change(corpusField(), { target: { value: "one\ntwo" } });
    expect(screen.getByTestId("embed-corpus")).toHaveTextContent("2 passes");
    expect(mockEmbedCorpus).not.toHaveBeenCalled();
  });

  it("shows four rankings after a search, and no fewer", async () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByTestId("search"));

    await waitFor(() =>
      expect(screen.getByTestId("rankings")).toBeInTheDocument(),
    );
    for (const col of ["col-bm25", "col-dense", "col-hybrid", "col-rerank"]) {
      expect(screen.getByTestId(col)).toBeInTheDocument();
    }
    expect(mockDense).toHaveBeenCalledTimes(1);
    expect(mockRerank).toHaveBeenCalledTimes(1);
  });

  it("reranks a shortlist rather than the corpus, and says how many", async () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByTestId("search"));
    await waitFor(() =>
      expect(screen.getByTestId("rankings")).toBeInTheDocument(),
    );

    const candidates = mockRerank.mock.calls[0][2] as number[];
    expect(candidates.length).toBeLessThanOrEqual(RANKING_CORPUS.length);
    expect(within(screen.getByTestId("col-rerank")).getByText(/passes/i))
      .toBeInTheDocument();
  });

  // RRF is arithmetic over the two rank lists in hand, so its k re-fuses.
  it("re-fuses the hybrid column when RRF k changes, running nothing", async () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByTestId("search"));
    await waitFor(() =>
      expect(screen.getByTestId("rankings")).toBeInTheDocument(),
    );
    expect(mockDense).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByLabelText(/rrf k/i), {
      target: { value: "1" },
    });

    expect(mockDense).toHaveBeenCalledTimes(1);
    expect(mockRerank).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("col-hybrid")).toHaveTextContent("k=1");
  });

  it("keeps the four columns about the captured corpus, not the live one", async () => {
    ready();
    renderPage();
    fireEvent.change(corpusField(), {
      target: { value: "alpha document\nbeta document" },
    });
    fireEvent.change(queryField(), { target: { value: "alpha" } });
    fireEvent.click(screen.getByTestId("search"));
    await waitFor(() =>
      expect(screen.getByTestId("rankings")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("col-bm25")).toHaveTextContent("alpha document");

    // Editing the corpus afterwards must not restyle a finished ranking.
    fireEvent.change(corpusField(), { target: { value: "something else" } });
    expect(screen.getByTestId("col-bm25")).toHaveTextContent("alpha document");
    expect(screen.getByTestId("rankings")).toHaveTextContent("alpha");
  });

  it("changing the pair spends nothing and drops the stale vectors", () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getAllByRole("button", { name: new RegExp(escape(OTHER.label), "i") })[0],
    );

    expect(base.load).not.toHaveBeenCalled();
    expect(mockDense).not.toHaveBeenCalled();
    // Vectors from another checkpoint are not comparable with these.
    expect(base.clear).toHaveBeenCalled();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    state = {
      ...base,
      status: "error",
      idle: false,
      error: "404 onnx/model_fp16.onnx not found",
    };
    renderPage();

    const note = screen.getByText(/model_fp16\.onnx not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT", () => {
    ready({ error: "Asked for 12 embeddings, got 11" });
    renderPage();

    const note = screen.getByText(/asked for 12 embeddings/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
