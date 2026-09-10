import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Pixels } from "@/vision/draw";

const toPngBlob = vi.fn();
vi.mock("@/vision/matte", () => ({
  toPngBlob: (...a: unknown[]) => toPngBlob(...a),
}));

const { DownloadImageButton } = await import("./DownloadImageButton");

const image: Pixels = {
  data: new Uint8ClampedArray(4 * 4 * 4),
  width: 4,
  height: 4,
  channels: 4,
};

describe("DownloadImageButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    URL.createObjectURL = vi.fn(() => "blob:cutout");
    URL.revokeObjectURL = vi.fn();
    toPngBlob.mockResolvedValue(new Blob(["png"], { type: "image/png" }));
  });

  it("hands back a PNG, the only common format that keeps alpha", async () => {
    // A "download the cut-out" button that produced a JPEG would hand back the
    // subject on an arbitrary opaque background — the one thing the matting
    // page exists to avoid.
    render(<DownloadImageButton image={image} filename="cutout.png" />);
    await waitFor(() =>
      expect(screen.getByTestId("download-png")).toHaveAttribute(
        "href",
        "blob:cutout",
      ),
    );
    expect(screen.getByTestId("download-png")).toHaveAttribute(
      "download",
      "cutout.png",
    );
  });

  it("says it is preparing until the blob exists, and is inert until then", async () => {
    // An `<a>` with no href is not a link and is not focusable, so it is already
    // inert; `aria-disabled` says so rather than relying on that.
    let release!: (blob: Blob) => void;
    toPngBlob.mockReturnValueOnce(
      new Promise<Blob>((resolve) => {
        release = resolve;
      }),
    );

    render(<DownloadImageButton image={image} filename="cutout.png" />);
    const link = screen.getByTestId("download-png");
    expect(link).not.toHaveAttribute("href");
    expect(link).toHaveAttribute("aria-disabled", "true");
    expect(link).toHaveTextContent(/preparing/i);

    release(new Blob(["png"], { type: "image/png" }));
    await waitFor(() => expect(link).toHaveAttribute("href", "blob:cutout"));
    expect(link).toHaveAttribute("aria-disabled", "false");
  });

  it("revokes the object URL on unmount", async () => {
    // The blob is the full-size image — megabytes — and one is created per
    // result, so leaving them alive pins every picture for the tab's lifetime.
    const view = render(<DownloadImageButton image={image} filename="a.png" />);
    await waitFor(() =>
      expect(screen.getByTestId("download-png")).toHaveAttribute("href"),
    );
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:cutout");
  });

  it("revokes the previous URL when the image changes", async () => {
    const view = render(<DownloadImageButton image={image} filename="a.png" />);
    await waitFor(() =>
      expect(screen.getByTestId("download-png")).toHaveAttribute("href"),
    );

    view.rerender(
      <DownloadImageButton image={{ ...image, width: 8 }} filename="a.png" />,
    );
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalled());
    expect(toPngBlob).toHaveBeenCalledTimes(2);
  });

  it("stays inert rather than throwing when the canvas has no context", async () => {
    // happy-dom, and a tab the browser has starved of memory. `toPngBlob`
    // resolves null there; an OUTPUT action that threw would take the slot down.
    toPngBlob.mockResolvedValueOnce(null);
    render(<DownloadImageButton image={image} filename="a.png" />);
    await waitFor(() =>
      expect(screen.getByTestId("download-png")).toHaveAttribute(
        "aria-disabled",
        "true",
      ),
    );
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it("renders an anchor, not a button", async () => {
    // Base UI's `Button` assumes a native <button> unless told otherwise and
    // warns at runtime when `render` supplies anything else — a download needs
    // an `<a download>`, so `nativeButton={false}` is required, not optional.
    render(<DownloadImageButton image={image} filename="a.png" />);
    await waitFor(() =>
      expect(screen.getByTestId("download-png").tagName).toBe("A"),
    );
  });
});
