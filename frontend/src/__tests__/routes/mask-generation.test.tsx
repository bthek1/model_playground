import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseSamResult } from "@/hooks/useSam";

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
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

const MASKS = [
  { data: new Uint8Array([1, 1, 0, 0]), width: 2, height: 2, score: 0.93 },
  { data: new Uint8Array([1, 1, 1, 1]), width: 2, height: 2, score: 0.51 },
  { data: new Uint8Array([1, 0, 0, 0]), width: 2, height: 2, score: 0.32 },
];

const mockRun = vi.fn().mockResolvedValue({ masks: MASKS, ms: 12 });
const mockEncode = vi.fn().mockResolvedValue(undefined);
const mockReset = vi.fn();

const baseState: UseSamResult = {
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
  encoding: false,
  encoded: false,
  encodedInMs: null,
  encode: mockEncode,
  run: mockRun,
  reset: mockReset,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseSamResult = { ...baseState };
const useSam = vi.fn(() => mockState);

vi.mock("@/hooks/useSam", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useSam: (...a: unknown[]) => useSam(...(a as [])) };
});

const { Route } = await import("@/routes/mask-generation");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  return render(<MaskPage />);
}

/** The route component, or a loud failure — so tests can render it as JSX. */
function MaskPage() {
  if (!Page) throw new Error("Mask generation route component not found");
  return <Page />;
}

const ready = (extra: Partial<UseSamResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

beforeEach(() => {
  vi.clearAllMocks();
  mockState = { ...baseState };
  mockRun.mockResolvedValue({ masks: MASKS, ms: 12 });
  mockEncode.mockResolvedValue(undefined);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

async function pickSample() {
  fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
  await waitFor(() => expect(fromUrl).toHaveBeenCalled());
}

/**
 * The one RUN action: "Generate mask" encodes the picture if it isn't encoded
 * yet, then decodes the points. Encoding used to fire off an effect the moment
 * a model and a picture coexisted, and each click on the canvas decoded
 * immediately — two inferences nobody pressed a button for.
 */
/**
 * Place the centre point — the keyboard route to a query.
 *
 * Waits for *enabled*, not merely present: the button exists from the first
 * render and only lights up once the capped frame lands, and `fireEvent.click`
 * on a disabled button is a silent no-op.
 */
async function placeCentrePoint() {
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: /point at the centre/i }),
    ).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: /point at the centre/i }));
}

async function generate() {
  // Re-queried inside the `waitFor`, never captured before it: a React
  // re-render replaces the DOM node, and polling the detached original is a
  // flake that never resolves.
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: /generate mask/i }),
    ).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: /generate mask/i }));
}

describe("MaskGenerationPage", () => {
  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /mask generation/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /slimsam-77/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /sam 2\.1 tiny/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useSam).toHaveBeenCalledWith("Xenova/slimsam-77-uniform");
    expect(baseState.load).not.toHaveBeenCalled();
    expect(mockEncode).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run controls disabled until an image is encoded", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /point at the centre/i }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: /clear points/i })).toBeDisabled();
  });

  it("shows the picture before a model exists, and encodes nothing", async () => {
    renderPage();
    await pickSample();
    await waitFor(() =>
      expect(screen.getByTestId("point-canvas")).toBeInTheDocument(),
    );
    expect(mockEncode).not.toHaveBeenCalled();
  });

  it("shows a picked image on a capped frame, and encodes nothing yet", async () => {
    mockState = ready();
    renderPage();
    await pickSample();

    // Capping the resolution is arithmetic on pixels — it is what makes the
    // picture clickable — so it happens on the pick. Both of SAM's graphs wait.
    await waitFor(() => expect(downscale).toHaveBeenCalledWith(fakeImage, 640));
    await screen.findByTestId("point-canvas");
    expect(mockEncode).not.toHaveBeenCalled();
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("encodes on the first Generate, on a capped frame", async () => {
    mockState = ready();
    renderPage();
    await pickSample();
    await placeCentrePoint();
    await generate();

    await waitFor(() => expect(mockEncode).toHaveBeenCalledTimes(1));
    // The token carries the image's identity *and* its size, so a re-render
    // reuses the embedding while a different picture does not.
    expect(mockEncode.mock.calls[0][0]).toBe("City street:8x8");
  });

  it("shows the encoding state, distinct from the load state", async () => {
    // Picked first, then re-rendered mid-encode: while `encoding` is true the
    // sample buttons are correctly disabled, so the picture has to exist before
    // the encode starts — which is also the real sequence.
    mockState = ready();
    const { rerender } = renderPage();
    await pickSample();
    await waitFor(() =>
      expect(screen.getByTestId("point-canvas")).toBeInTheDocument(),
    );

    mockState = ready({ encoding: true });
    rerender(<MaskPage />);

    expect(screen.getByTestId("encoding")).toBeInTheDocument();
    // The LOAD slot is not what is busy — the model is loaded.
    expect(screen.getByTestId("slot-3")).toContainElement(
      screen.getByTestId("encoding"),
    );
    expect(screen.getByTestId("slot-2")).not.toContainElement(
      screen.getByTestId("encoding"),
    );
  });

  it("says how long the encode took, once it is done", async () => {
    mockState = ready({ encoded: true, encodedInMs: 380 });
    renderPage();
    await pickSample();
    await waitFor(() =>
      expect(screen.getByTestId("encoded")).toHaveTextContent(/380 ms/),
    );
  });

  it("reads a click on the picture as a positive point, and decodes on request", async () => {
    mockState = ready({ encoded: true });
    renderPage();
    await pickSample();

    const canvas = await screen.findByTestId("point-canvas");
    fireEvent.click(canvas, { clientX: 4, clientY: 4 });
    // Placing a point is input. Decoding on every click made the canvas a
    // hidden RUN trigger, and a mis-aimed click cost an inference.
    expect(mockRun).not.toHaveBeenCalled();

    await generate();
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toEqual([
      { x: expect.any(Number), y: expect.any(Number), positive: true },
    ]);
    expect(mockRun.mock.calls[0][0][0].positive).toBe(true);
  });

  it("reads an alt-click as a negative point", async () => {
    // 1 means "the object", 0 means "not the object". Swapping them inverts
    // every mask a second click was meant to refine.
    mockState = ready({ encoded: true });
    renderPage();
    await pickSample();

    const canvas = await screen.findByTestId("point-canvas");
    fireEvent.click(canvas, { clientX: 4, clientY: 4, altKey: true });
    await generate();

    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0][0].positive).toBe(false);
  });

  it("keeps Generate disabled until there is a point to decode", async () => {
    mockState = ready({ encoded: false });
    renderPage();
    await pickSample();
    await screen.findByTestId("point-canvas");

    // A picture alone is not a query: SAM needs somewhere to look.
    expect(
      screen.getByRole("button", { name: /generate mask/i }),
    ).toBeDisabled();
    expect(mockRun).not.toHaveBeenCalled();
    expect(mockEncode).not.toHaveBeenCalled();
  });

  it("offers a keyboard route to a mask", async () => {
    // Clicking a canvas is inherently a pointer gesture, and a page whose only
    // input is a click is a page some people cannot use at all.
    mockState = ready({ encoded: true });
    renderPage();
    await pickSample();

    await placeCentrePoint();
    await generate();

    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toEqual([{ x: 4, y: 4, positive: true }]);
  });

  it("accumulates points, and lists them", async () => {
    mockState = ready({ encoded: true });
    renderPage();
    await pickSample();

    const canvas = await screen.findByTestId("point-canvas");
    fireEvent.click(canvas, { clientX: 2, clientY: 2 });
    fireEvent.click(canvas, { clientX: 6, clientY: 6, altKey: true });

    await waitFor(() =>
      expect(
        within(screen.getByTestId("points")).getAllByRole("listitem"),
      ).toHaveLength(2),
    );
    // Two points, one decode. Refining is now "place the points you want, then
    // ask" rather than an inference per click.
    await generate();
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun.mock.calls[0][0]).toHaveLength(2);
  });

  it("clears the points and the embedding-backed state together", async () => {
    mockState = ready({ encoded: true });
    renderPage();
    await pickSample();

    fireEvent.click(await screen.findByTestId("point-canvas"), {
      clientX: 2,
      clientY: 2,
    });
    await waitFor(() => expect(screen.getByTestId("points")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /clear points/i }));
    expect(screen.queryByTestId("points")).not.toBeInTheDocument();
    expect(mockReset).toHaveBeenCalled();
  });

  it("offers all three candidates, and switches between them", async () => {
    // A point is ambiguous by construction — the wheel, the door, or the whole
    // car — and showing only the top-scoring mask hides the most interesting
    // thing about the model.
    mockState = ready({ encoded: true, result: { masks: MASKS, ms: 12 } });
    renderPage();
    await pickSample();

    await waitFor(() =>
      expect(screen.getByTestId("mask-canvas")).toBeInTheDocument(),
    );
    const out = within(screen.getByTestId("slot-4"));
    // By role: each score also appears in the facts line below, so a bare text
    // query matches twice.
    expect(out.getByRole("button", { name: /^1 · 0\.930$/ })).toBeInTheDocument();
    expect(out.getByRole("button", { name: /^2 · 0\.510$/ })).toBeInTheDocument();
    expect(out.getByRole("button", { name: /^3 · 0\.320$/ })).toBeInTheDocument();

    // The first candidate covers half the 2x2 mask; the second covers all of it.
    expect(screen.getByTestId("mask-facts")).toHaveTextContent(/50\.0%/);
    fireEvent.click(out.getByRole("button", { name: /^2 · 0\.510$/ }));
    expect(screen.getByTestId("mask-facts")).toHaveTextContent(/100\.0%/);
  });

  it("shows the decode time, because sub-100 ms is the claim", async () => {
    mockState = ready({ encoded: true, result: { masks: MASKS, ms: 12 } });
    renderPage();
    await pickSample();
    await waitFor(() =>
      expect(screen.getByTestId("decode-ms")).toHaveTextContent(/decode 12 ms/),
    );
  });

  it("surfaces a failed decode of the picture in the RUN slot", async () => {
    mockState = ready();
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    const note = await screen.findByText(/unsupported image type/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "404 not found",
    };
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
