import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const fakeImage = { width: 4, height: 4, channels: 3 } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
}));

const { useImagePick } = await import("./useImagePick");

const sample = {
  id: "tiger",
  label: "Tiger",
  url: "https://example.test/tiger.jpg",
  hint: "",
  expect: "",
};

describe("useImagePick", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fromFile.mockResolvedValue(fakeImage);
    fromUrl.mockResolvedValue(fakeImage);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("decodes a file and owns the object URL it created", async () => {
    const { result } = renderHook(() => useImagePick());
    const file = new File(["x"], "cat.png", { type: "image/png" });

    act(() => result.current.pickFile(file));
    await waitFor(() => expect(result.current.picked).not.toBeNull());

    expect(fromFile).toHaveBeenCalledWith(file);
    expect(result.current.picked).toMatchObject({
      previewUrl: "blob:preview",
      owned: true,
      name: "cat.png",
    });
  });

  it("frees the previous object URL when a second image is picked", async () => {
    // A preview URL that outlives its <img> pins the decoded bitmap in memory
    // for the tab's lifetime.
    const { result } = renderHook(() => useImagePick());
    act(() => result.current.pickFile(new File(["x"], "a.png")));
    await waitFor(() => expect(result.current.picked).not.toBeNull());

    act(() => result.current.pickSample(sample));
    await waitFor(() =>
      expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview"),
    );
  });

  it("does not revoke a sample URL it never created", async () => {
    const { result } = renderHook(() => useImagePick());
    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.picked).not.toBeNull());

    act(() => result.current.clear());
    expect(URL.revokeObjectURL).not.toHaveBeenCalled();
  });

  it("frees the object URL on unmount", async () => {
    const { result, unmount } = renderHook(() => useImagePick());
    act(() => result.current.pickFile(new File(["x"], "a.png")));
    await waitFor(() => expect(result.current.picked).not.toBeNull());

    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview");
  });

  it("reports a failed decode, and picks nothing", async () => {
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    const { result } = renderHook(() => useImagePick());

    act(() => result.current.pickSample(sample));
    await waitFor(() =>
      expect(result.current.error).toMatch(/unsupported image type/i),
    );
    expect(result.current.picked).toBeNull();
  });

  // The hook used to take an `onPicked` callback, and every vision route used
  // it to run the model on the freshly decoded image — which turned the sample
  // row and the file dialog into hidden RUN triggers. Picking is input; the
  // route's button is the only trigger (model-page-pattern.md §1.6).
  it("takes no run callback at all — picking cannot start an inference", () => {
    // Typed as taking no arguments, so this is the runtime half of the same
    // guarantee: anything handed in is ignored rather than quietly invoked.
    const onPicked = vi.fn();
    const { result } = renderHook(() =>
      (useImagePick as (o?: unknown) => ReturnType<typeof useImagePick>)({
        onPicked,
      }),
    );

    act(() => result.current.pickSample(sample));
    expect(onPicked).not.toHaveBeenCalled();
  });

  it("ignores an empty file selection rather than clearing the preview", async () => {
    const { result } = renderHook(() => useImagePick());
    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.picked).not.toBeNull());

    act(() => result.current.pickFile(undefined));
    expect(result.current.picked).not.toBeNull();
    expect(fromFile).not.toHaveBeenCalled();
  });
});
