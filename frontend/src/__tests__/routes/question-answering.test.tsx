import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseQaResult } from "@/hooks/useQa";
import type { QaAnswer } from "@/text/qa/types";

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

const DISTILBERT = "Xenova/distilbert-base-cased-distilled-squad";

const EIFFEL =
  "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris. It stood as the world's tallest man-made structure for 41 years, until the Chrysler Building in New York was finished in 1930.";

/** The real answer for the first sample, measured against the q8 build. */
const GUSTAVE: QaAnswer = {
  text: "Gustave Eiffel",
  decoded: "Gustave Eiffel",
  score: 0.996898,
  start: 30,
  end: 44,
  truncated: false,
};

const mockRun = vi.fn<UseQaResult["run"]>().mockResolvedValue(GUSTAVE);
const base: UseQaResult = {
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

let state: UseQaResult = { ...base };
const useQa = vi.fn(() => state);
vi.mock("@/hooks/useQa", () => ({ useQa: () => useQa() }));

// The cache probe, so the "cached weights still do not load themselves"
// assertion below has something to probe.
let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: () => Promise.resolve(),
}));

const { Route } = await import("@/routes/question-answering");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Question-answering route component not found");
  render(<Page />);
}

const ready = (extra: Partial<UseQaResult> = {}): UseQaResult => ({
  ...base,
  status: "ready",
  idle: false,
  ready: true,
  ...extra,
});

const passage = () => screen.getByLabelText(/^passage$/i);
const questionBox = () => screen.getByLabelText(/^question$/i);
const trigger = () => screen.getByRole("button", { name: /^answer$/i });

describe("QuestionAnsweringPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    cached = new Set();
    mockRun.mockResolvedValue(GUSTAVE);
  });

  it("renders the heading and the single catalogue entry", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /question answering/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /distilbert squad/i, pressed: true }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves: the hook is handed **no auto-load argument at all** (its
    // default is `idle`), and nothing has called load() behind the user's back.
    expect(useQa).toHaveBeenCalled();
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

  it("gates the trigger on `ready`, but never the input fields", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    // Writing the passage and the question first is free and commits to
    // nothing (model-page-pattern.md §1.2).
    expect(passage()).toBeEnabled();
    expect(questionBox()).toBeEnabled();
  });

  // The core of the SELECT → LOAD → INPUT → GENERATE contract, and the single
  // most valuable assertion here: an input that silently starts an inference
  // looks exactly like a working page.
  it("runs nothing while either field is edited, even with a model ready", () => {
    state = ready();
    renderPage();

    fireEvent.change(passage(), { target: { value: "A brand new passage." } });
    fireEvent.change(questionBox(), { target: { value: "What is new?" } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills both boxes", () => {
    state = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /a fact in the passage/i }));

    expect(passage()).toHaveValue(EIFFEL);
    expect(questionBox()).toHaveValue("Who built the Eiffel Tower?");
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("answers when — and only when — the trigger is pressed", async () => {
    state = ready();
    renderPage();

    fireEvent.change(questionBox(), { target: { value: "Who built it?" } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // Question first, passage second — the order the engine expects. Swapping
    // them produces a fluent answer to the wrong question, with no error.
    expect(mockRun).toHaveBeenCalledWith("Who built it?", EIFFEL);
  });

  it("refuses to run on an empty passage or an empty question", () => {
    state = ready();
    renderPage();

    fireEvent.change(questionBox(), { target: { value: "   " } });
    expect(trigger()).toBeDisabled();

    fireEvent.change(questionBox(), { target: { value: "Who built it?" } });
    fireEvent.change(passage(), { target: { value: "  \n " } });
    expect(trigger()).toBeDisabled();
  });

  it("marks the answer in the passage at the offsets the model returned", async () => {
    state = ready();
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("answer-span")).toBeInTheDocument(),
    );
    const mark = screen.getByTestId("span-mark");
    expect(mark).toHaveTextContent("Gustave Eiffel");
    // The range itself, not only the words: on a passage that names someone
    // twice, the right string at the wrong occurrence looks correct.
    const range = screen.getByTestId("answer-range");
    expect(range).toHaveAttribute("data-start", "30");
    expect(range).toHaveAttribute("data-end", "44");
    expect(EIFFEL.slice(30, 44)).toBe("Gustave Eiffel");
  });

  it("shows the score beside every answer", async () => {
    state = ready();
    renderPage();
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("answer-score")).toHaveTextContent("0.997"),
    );
  });

  // The whole passage is re-rendered from the captured string, so a stale
  // render is worse here than a stale label: the offsets would be applied to a
  // string they were not computed against and the mark would move.
  it("renders the passage captured inside the run, not the textarea's value", async () => {
    state = ready();
    renderPage();
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("answer-span")).toBeInTheDocument(),
    );

    fireEvent.change(passage(), { target: { value: "Something else entirely." } });
    fireEvent.change(questionBox(), { target: { value: "A different question?" } });

    expect(screen.getByTestId("answer-span")).toHaveTextContent(
      /world's tallest man-made structure/i,
    );
    expect(screen.getByTestId("span-mark")).toHaveTextContent("Gustave Eiffel");
    expect(screen.getByTestId("ran-question")).toHaveTextContent(
      "Who built the Eiffel Tower?",
    );
  });

  // The `/video-classification` precedent: a standing limitation the page must
  // state, pinned so it cannot be deleted as decoration. It is in OUTPUT's
  // description rather than beside the result, so it is on screen *before* the
  // first answer too — a caveat that appears only once you already believe the
  // answer has arrived too late.
  it("states that the model cannot abstain, before any run and after one", async () => {
    state = ready();
    renderPage();

    const note = screen.getByTestId("no-abstain-note");
    expect(note).toHaveTextContent(/cannot say/i);
    expect(note).toHaveTextContent(/SQuAD 1\.1/);
    expect(note).toHaveTextContent(/always returns a span/i);
    // And it says why it cannot simply be fixed with a better checkpoint.
    expect(note).toHaveTextContent(/no ONNX export/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);

    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("answer-span")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("no-abstain-note")).toBeInTheDocument();
  });

  it("ships a sample whose question its passage cannot answer", () => {
    renderPage();
    const sample = screen.getByTestId("sample-unanswerable");
    expect(sample).toBeInTheDocument();
    // Labelled as such on the button itself — the demonstration only works if
    // the user knows the passage has no answer before the model gives one.
    expect(sample).toHaveTextContent(/no answer/i);
  });

  it("quotes the answer rather than mis-marking it when alignment failed", async () => {
    state = ready();
    mockRun.mockResolvedValue({
      text: "",
      decoded: "Gustave Eiffel",
      score: 0.83,
      start: null,
      end: null,
      truncated: false,
    });
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("answer-unaligned")).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("answer-span")).not.toBeInTheDocument();
    expect(screen.getByTestId("answer-unaligned")).toHaveTextContent(
      "Gustave Eiffel",
    );
  });

  it("says when the passage ran past the model's window", async () => {
    state = ready();
    mockRun.mockResolvedValue({ ...GUSTAVE, truncated: true });
    renderPage();
    fireEvent.click(trigger());

    await waitFor(() =>
      expect(screen.getByTestId("answer-truncated")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("answer-truncated")).toHaveTextContent(/512/);
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

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    state = ready({ error: "There is no passage to answer from." });
    renderPage();

    const note = screen.getByText(/no passage to answer from/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});

describe("QuestionAnsweringPage — after a page refresh", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    cached = new Set();
  });

  // A Worker cannot outlive a page load, so a refresh always lands on `idle`.
  // A cache hit makes the LOAD click *cheap* — it does not make it unnecessary,
  // and the button says so rather than disappearing.
  it("stays idle with the weights already cached, and says they are", async () => {
    cached = new Set([DISTILBERT]);
    renderPage();

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /load model \(cached\)/i }),
      ).toBeEnabled(),
    );
    expect(base.load).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });
});
