import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseNerResult } from "@/hooks/useNer";
import type { EntitySpan } from "@/text/highlight";

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

// Offsets into NER_SAMPLES[0]: "Priya Raman flew from Wellington to Berlin …"
const SPANS: EntitySpan[] = [
  { start: 0, end: 11, label: "PER", score: 0.99 },
  { start: 22, end: 32, label: "LOC", score: 0.98 },
  { start: 36, end: 42, label: "LOC", score: 0.97 },
  { start: 73, end: 80, label: "ORG", score: 0.95 },
];

const mockRun = vi.fn().mockResolvedValue(SPANS);
const base: UseNerResult = {
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
  unplaced: [],
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseNerResult = { ...base };
let writeText: ReturnType<typeof vi.fn>;
const useNer = vi.fn(() => mockState);

vi.mock("@/hooks/useNer", () => ({
  useNer: (...args: unknown[]) => useNer(...(args as [])),
}));

const { Route } = await import("@/routes/token-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Token classification route component not found");
  render(<Page />);
}

const ready = (extra: Partial<UseNerResult> = {}) => ({
  ...base,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

/** Load, tag, and wait for the overlay. */
async function tagged() {
  mockState = ready();
  renderPage();
  fireEvent.click(screen.getByRole("button", { name: /find entities/i }));
  await waitFor(() =>
    expect(screen.getByTestId("span-overlay")).toBeInTheDocument(),
  );
}

describe("TokenClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...base };
    // happy-dom exposes `navigator.clipboard` as a getter-only property.
    writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
  });

  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /token classification/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /bert base ner/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /mbert ner/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useNer).toHaveBeenCalledWith("Xenova/bert-base-NER");
    expect(base.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^load model$/i }));
    expect(base.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    for (const step of [1, 2, 3, 4]) {
      expect(screen.getByTestId(`slot-${step}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the trigger disabled until ready, but never the textarea", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /find entities/i }),
    ).toBeDisabled();
    expect(screen.getByLabelText(/text to tag/i)).toBeEnabled();
  });

  it("runs nothing while typing or picking a sample", () => {
    mockState = ready();
    renderPage();

    fireEvent.change(screen.getByLabelText(/text to tag/i), {
      target: { value: "Ada Lovelace worked in London." },
    });
    fireEvent.click(screen.getByRole("button", { name: /a news lede/i }));

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("tags only when the trigger is pressed", async () => {
    mockState = ready();
    renderPage();
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /find entities/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
  });

  it("marks the entities in the text captured inside the run", async () => {
    await tagged();
    const marks = screen.getAllByTestId("span-mark");
    expect(marks).toHaveLength(4);
    expect(marks[0]).toHaveTextContent("Priya Raman");
    expect(marks[0]).toHaveTextContent("PER");

    // Editing the box afterwards must not restyle a finished result.
    fireEvent.change(screen.getByLabelText(/text to tag/i), {
      target: { value: "something else entirely" },
    });
    expect(screen.getAllByTestId("span-mark")[0]).toHaveTextContent(
      "Priya Raman",
    );
  });

  // The rule this page shares with /vad's threshold and detection's floor: a
  // control that re-reads a result already in hand spends nothing.
  it("redacts by re-deriving, calling the model zero more times", async () => {
    await tagged();
    expect(mockRun).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByTestId("redact-toggle"));
    await waitFor(() =>
      expect(screen.getAllByTestId("span-redacted").length).toBeGreaterThan(0),
    );
    expect(mockRun).toHaveBeenCalledTimes(1);

    // And so does changing *which* types are redacted.
    fireEvent.click(
      screen.getByTestId("redact-types").querySelector("button")!,
    );
    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("redacts PER and LOC by default and leaves ORG alone", async () => {
    await tagged();
    fireEvent.click(screen.getByTestId("redact-toggle"));

    await waitFor(() =>
      expect(screen.getAllByTestId("span-redacted")).toHaveLength(3),
    );
    // The organisation is usually the part worth keeping, which is why the
    // choice is per-type rather than all-or-nothing.
    expect(screen.getAllByTestId("span-mark")).toHaveLength(1);
    expect(screen.getAllByTestId("span-mark")[0]).toHaveTextContent("ORG");
  });

  it("offers the redacted text as a copy once redaction is on", async () => {
    await tagged();
    expect(
      screen.queryByRole("button", { name: /copy redacted/i }),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("redact-toggle"));
    const copy = await screen.findByRole("button", { name: /copy redacted/i });
    fireEvent.click(copy);

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(
        expect.stringContaining("[PER]"),
      ),
    );
    // The organisation survived the redaction, so it is still in the copy.
    expect(writeText.mock.calls[0][0]).toContain("Siemens");
  });

  it("says so when the model found nothing, rather than showing an empty panel", async () => {
    mockState = ready({ run: vi.fn().mockResolvedValue([]) });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /find entities/i }));

    expect(
      await screen.findByText(/found nothing to tag/i),
    ).toBeInTheDocument();
    // And there is nothing to redact, so no toggle.
    expect(screen.queryByTestId("redact-toggle")).not.toBeInTheDocument();
  });

  it("names the entity types the selected head can emit", () => {
    renderPage();
    expect(screen.getByText(/PER, ORG, LOC, MISC/)).toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = {
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
    mockState = ready({ error: "Input is too long" });
    renderPage();
    const note = screen.getByText(/input is too long/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
