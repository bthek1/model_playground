import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseZeroShotImageResult } from "@/hooks/useZeroShotImage";

const fakeImage = { width: 4, height: 4, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
  fromVideo: vi.fn(),
  openCamera: vi.fn().mockResolvedValue(() => {}),
}));

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

const RESULT = [
  {
    template: "{}",
    textCached: false,
    textMs: 14,
    imageMs: 31,
    scores: [
      { label: "cat", score: 0.61 },
      { label: "dog", score: 0.3 },
      { label: "empty sofa", score: 0.09 },
    ],
  },
  {
    template: "a photo of a {}",
    textCached: true,
    textMs: 0,
    imageMs: 29,
    scores: [
      { label: "cat", score: 0.82 },
      { label: "dog", score: 0.14 },
      { label: "empty sofa", score: 0.04 },
    ],
  },
];

const mockRun = vi.fn().mockResolvedValue(RESULT);
const baseState: UseZeroShotImageResult = {
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
let mockState: UseZeroShotImageResult = { ...baseState };
const useZeroShotImage = vi.fn(() => mockState);

vi.mock("@/hooks/useZeroShotImage", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useZeroShotImage: (...a: unknown[]) => useZeroShotImage(...(a as [])),
  };
});

const { Route } = await import("@/routes/zero-shot-image-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Zero-shot route component not found");
  render(<Page />);
}

const RUN_BUTTON = /score labels/i;

// The RUN trigger. Picking an input never starts an inference any more
// (model-page-pattern.md §1.6), so a test that wants a result asks for one.
async function pickAndRun(sample: RegExp) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
}

const ready = (extra: Partial<UseZeroShotImageResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("ZeroShotImageClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(RESULT);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /zero-shot image classification/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /clip vit-b\/32/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /siglip 2/i })).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useZeroShotImage).toHaveBeenCalledWith("Xenova/clip-vit-base-patch32");
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /score labels/i })).toBeDisabled();
  });

  it("starts from a label set the user can edit", () => {
    renderPage();
    const run = within(screen.getByTestId("slot-3"));
    expect(run.getByText("cat")).toBeInTheDocument();

    fireEvent.click(run.getByRole("button", { name: /remove dog/i }));
    expect(run.queryByText("dog")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/^labels$/i), {
      target: { value: "a bicycle" },
    });
    fireEvent.click(run.getByRole("button", { name: /^add$/i }));
    expect(run.getByText("a bicycle")).toBeInTheDocument();
  });

  it("scores both templates on a still image", async () => {
    // The page's whole thesis: `a photo of a {}` against a bare `{}`, side by
    // side, because a before/after the user has to remember is not a demo.
    mockState = ready();
    renderPage();

    await pickAndRun(/^cats$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    expect(mockRun).toHaveBeenCalledWith(
      fakeImage,
      ["cat", "dog", "empty sofa"],
      ["{}", "a photo of a {}"],
    );
  });

  it("sends only the chosen template once the comparison is turned off", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByLabelText(/also score the bare/i));
    await pickAndRun(/^cats$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    expect(mockRun.mock.calls[0][2]).toEqual(["a photo of a {}"]);
  });

  it("changes the string it sends when the template changes", async () => {
    // Asserted on the call, not on the score: the effect is the model's
    // business, the prompt is ours.
    mockState = ready();
    renderPage();

    fireEvent.change(screen.getByLabelText(/prompt template/i), {
      target: { value: "a blurry photo of a {}" },
    });
    await pickAndRun(/^cats$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    expect(mockRun.mock.calls[0][2]).toContain("a blurry photo of a {}");
  });

  it("shows one column per template, in the user's own words", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/^cats$/i);

    const out = within(screen.getByTestId("slot-4"));
    await waitFor(() =>
      expect(out.getByRole("columnheader", { name: "{}" })).toBeInTheDocument(),
    );
    expect(
      out.getByRole("columnheader", { name: "a photo of a {}" }),
    ).toBeInTheDocument();

    // One row per label, in the order the user wrote them — not re-sorted per
    // column, which would make the comparison unreadable.
    const rows = out.getAllByRole("row").slice(1);
    expect(rows.map((r) => within(r).getAllByRole("cell")[0].textContent)).toEqual(
      ["cat", "dog", "empty sofa"],
    );
    expect(within(rows[0]).getAllByRole("cell").map((c) => c.textContent)).toEqual(
      ["cat", "0.610", "0.820"],
    );
    expect(out.getByTestId("template-verdict")).toBeInTheDocument();
  });

  it("shows what the label cache saved, per template", async () => {
    // An optimisation nobody can see is an optimisation nobody can check: the
    // text tower is ~40% of CLIP's work and is skipped entirely on a cache hit.
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/^cats$/i);

    const out = within(screen.getByTestId("slot-4"));
    await waitFor(() =>
      expect(out.getByTestId("encode-cost")).toBeInTheDocument(),
    );
    const cost = out.getByTestId("encode-cost");
    expect(cost).toHaveTextContent(/labels\s*14 ms/); // first template: a miss
    expect(cost).toHaveTextContent(/reused/); // second: served from the cache
    expect(cost).toHaveTextContent(/image\s*31 ms/);
  });

  it("says a zero-shot score is relative to the labels given", async () => {
    // "cat 0.98" against ["cat", "dog"] means "more cat than dog", not "a cat
    // is present" — and the model has no way to say "none of these".
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/^cats$/i);

    const out = within(screen.getByTestId("slot-4"));
    await waitFor(() =>
      expect(out.getByText(/relative to the labels/i)).toBeInTheDocument(),
    );
    expect(out.getByText(/none of these/i)).toBeInTheDocument();
  });

  it("surfaces a failed decode in the RUN slot, not the output", async () => {
    mockState = ready();
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    const note = await screen.findByText(/unsupported image type/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = { ...baseState, status: "error", idle: false, error: "404 not found" };
    renderPage();
    const note = screen.getByText(/404 not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "Non-zero status code" });
    renderPage();
    const note = screen.getByText(/non-zero status code/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
  });
});
