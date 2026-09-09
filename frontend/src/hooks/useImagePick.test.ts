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

  it("keeps an inference failure out of its own error slot", async () => {
    // A failed run belongs to OUTPUT, where the task hook already reports it.
    // Surfacing it here too would put the same message in two slots at once.
    const onPicked = vi.fn().mockRejectedValue(new Error("session failed"));
    const { result } = renderHook(() => useImagePick({ onPicked }));

    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(onPicked).toHaveBeenCalled());
    expect(result.current.error).toBeNull();
    expect(result.current.picked).not.toBeNull();
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
