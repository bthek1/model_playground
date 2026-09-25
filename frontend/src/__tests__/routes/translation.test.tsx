import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseTranslateResult } from "@/hooks/useTranslate";
import { TRANSLATION_MODELS } from "@/text/catalogue";

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

const EN_DE = TRANSLATION_MODELS[0];
const DE_EN = TRANSLATION_MODELS[1];

const mockRun = vi.fn(async () => "Die Besprechung wurde verlegt.");
const base: UseTranslateResult = {
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
  meta: EN_DE,
  result: null,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};

let state: UseTranslateResult = { ...base };
const useTranslate = vi.fn((_id: string) => state);
vi.mock("@/hooks/useTranslate", () => ({
  useTranslate: (...args: unknown[]) => useTranslate(...(args as [string])),
}));

const { Route } = await import("@/routes/translation");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Translation route component not found");
  render(<Page />);
}

// By role, not by label: the RUN band is a `region` labelled "Source text" too,
// so `getByLabelText` matches both the band and the field inside it (the trap
// CLAUDE.md names for every route test).
const box = () => screen.getByRole("textbox");
const trigger = () => screen.getByRole("button", { name: /^translate$/i });

function ready(extra: Partial<UseTranslateResult> = {}) {
  state = { ...base, status: "ready", idle: false, ready: true, ...extra };
}

describe("TranslationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    state = { ...base };
    localStorage.clear();
  });

  it("renders the heading and every direction", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { level: 1, name: /translation/i }),
    ).toBeInTheDocument();
    for (const m of TRANSLATION_MODELS) {
      expect(
        screen.getAllByRole("button", { name: new RegExp(m.label, "i") }).length,
      ).toBeGreaterThan(0);
    }
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useTranslate).toHaveBeenCalledWith(EN_DE.id);
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

  // **The rule this page is most likely to break.** A direction is a
  // checkpoint, so the control is SELECT — and SELECT never loads.
  it("changing the direction does not load anything", () => {
    ready();
    renderPage();
    fireEvent.click(
      screen.getAllByRole("button", { name: new RegExp(DE_EN.label, "i") })[0],
    );

    expect(base.load).not.toHaveBeenCalled();
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("says a direction change is a download, next to the control", () => {
    renderPage();
    expect(screen.getByTestId("pair-is-a-download")).toHaveTextContent(
      /separate checkpoint/i,
    );
    expect(screen.getByTestId("pair-is-a-download")).toHaveTextContent(
      /rather than a setting/i,
    );
  });

  // The trade-off in numbers rather than prose, and the multilingual model is
  // quoted without being offered — it is over the app's size bar.
  it("quotes the specialists-versus-generalist trade-off", () => {
    renderPage();
    const note = screen.getByTestId("specialists-note");
    expect(note).toHaveTextContent(/NLLB-200-distilled-600M/i);
    // NLLB's fp16 size. The roadmap's 894.6 MB is its q8 figure, and q8 is not
    // what a browser loads (§1.1's finding, again).
    expect(note).toHaveTextContent(/1\.6 GB/);
    // The comparison is **per pair**, not the whole catalogue: nobody downloads
    // all six directions, and summing them comes to 1.6 GB too — a number that
    // reads as an argument against the design rather than for it.
    expect(note).toHaveTextContent(/\b8x\b/);
    expect(note).toHaveTextContent(/not offered/i);
  });

  // An inverted download is unusual enough that leaving it unexplained reads as
  // a bug in the page, and the explanation is a measurement.
  it("explains why the CPU download is the larger one", () => {
    renderPage();
    expect(screen.getByTestId("wasm-cost-note")).toHaveTextContent(
      /cannot be quantized/i,
    );
  });

  it("gates the trigger on ready, but never the textarea", () => {
    renderPage();
    expect(trigger()).toBeDisabled();
    expect(box()).toBeEnabled();
    expect(
      screen.getByText(/load a direction to translate/i),
    ).toBeInTheDocument();
  });

  it("runs nothing while the user types", () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "Something new." } });

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("runs nothing when a sample is picked — it only fills the box", () => {
    ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /an idiom/i }));

    expect(box()).toHaveValue(
      "They decided to bite the bullet and rewrite the whole thing from scratch.",
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("translates when — and only when — the trigger is pressed", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The meeting moved." } });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(trigger());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // No language arguments — the direction is the checkpoint.
    expect(mockRun).toHaveBeenCalledWith("The meeting moved.");
  });

  it("labels the output with the source captured at run time", async () => {
    ready();
    renderPage();
    fireEvent.change(box(), { target: { value: "The original source." } });
    fireEvent.click(trigger());
    await waitFor(() =>
      expect(screen.getByTestId("translated")).toBeInTheDocument(),
    );

    expect(screen.getByTestId("ran-source")).toHaveTextContent(
      "The original source.",
    );
    expect(screen.getByTestId("translation-text")).toHaveTextContent(
      "Die Besprechung wurde verlegt.",
    );

    // Editing afterwards cannot relabel a result already on screen.
    fireEvent.change(box(), { target: { value: "edited after the run" } });
    expect(screen.getByTestId("ran-source")).toHaveTextContent(
      "The original source.",
    );
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
    ready({ error: "Input is too long" });
    renderPage();

    const note = screen.getByText(/input is too long/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
