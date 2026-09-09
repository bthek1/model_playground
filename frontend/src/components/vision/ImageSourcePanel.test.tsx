import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

/** A typed no-op ref callback, for the cases that do not inspect the node. */
const noopRef = () => {};

import { ImageSourcePanel } from "./ImageSourcePanel";

const samples = [
  { id: "cats", label: "Cats", url: "u1", hint: "two cats", expect: "" },
  { id: "tiger", label: "Tiger", url: "u2", hint: "one tiger", expect: "" },
];

const picked = {
  image: {} as never,
  previewUrl: "blob:preview",
  owned: true,
  name: "cat.png",
};

function renderPanel(
  props: Partial<Parameters<typeof ImageSourcePanel>[0]> = {},
) {
  const onFile = vi.fn();
  const onSample = vi.fn();
  const view = render(
    <ImageSourcePanel
      picked={null}
      preparing={null}
      samples={samples}
      sampleHint="Samples"
      onFile={onFile}
      onSample={onSample}
      busy={false}
      {...props}
    />,
  );
  return { ...view, onFile, onSample };
}

describe("ImageSourcePanel", () => {
  it("prompts for an image before one is picked, and shows it after", () => {
    const { rerender } = renderPanel();
    expect(screen.getByText(/drop an image here/i)).toBeInTheDocument();

    rerender(
      <ImageSourcePanel
        picked={picked}
        preparing={null}
        samples={samples}
        sampleHint="Samples"
        onFile={vi.fn()}
        onSample={vi.fn()}
        busy={false}
      />,
    );
    expect(screen.getByAltText(/selected input: cat\.png/i)).toBeInTheDocument();
  });

  it("accepts a drop and a file selection", () => {
    const { onFile } = renderPanel();
    const file = new File(["x"], "dropped.png", { type: "image/png" });

    fireEvent.drop(screen.getByText(/drop an image here/i).parentElement!, {
      dataTransfer: { files: [file] },
    });
    expect(onFile).toHaveBeenCalledWith(file);

    fireEvent.change(screen.getByLabelText(/upload an image/i), {
      target: { files: [file] },
    });
    expect(onFile).toHaveBeenCalledTimes(2);
  });

  it("renders no camera controls on a route that has no live mode", () => {
    const { container } = renderPanel();
    expect(
      screen.queryByRole("button", { name: /use camera/i }),
    ).not.toBeInTheDocument();
    expect(container.querySelector("video")).toBeNull();
  });

  it("mounts the video before it is live, so the ref exists to attach a stream to", () => {
    // `useCamera` needs the element to exist before it can attach a stream, and
    // a ref to a node React has not created yet is null at exactly the wrong
    // moment. So the <video> is mounted and hidden, never conditionally rendered.
    const videoRef = vi.fn();
    const { container } = renderPanel({
      camera: { live: false, onToggle: vi.fn(), videoRef },
    });

    const video = container.querySelector("video");
    expect(video).not.toBeNull();
    expect(video).toHaveClass("hidden");
    expect(videoRef).toHaveBeenCalledWith(video);
  });

  it("swaps the still preview for the video once live, and locks the still inputs", () => {
    const { container } = renderPanel({
      picked,
      camera: { live: true, onToggle: vi.fn(), videoRef: noopRef, fps: 12 },
    });

    expect(container.querySelector("video")).not.toHaveClass("hidden");
    expect(screen.queryByAltText(/selected input/i)).not.toBeInTheDocument();

    // A sample or an upload mid-stream would fight the frame pump for the slot.
    expect(screen.getByRole("button", { name: /^Cats$/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload image/i })).toBeDisabled();
    // But stopping must always be possible.
    expect(screen.getByRole("button", { name: /stop camera/i })).toBeEnabled();
    expect(screen.getByTestId("live-fps")).toHaveTextContent("12 fps");
  });

  it("ignores a drop while the camera owns the preview", () => {
    const { onFile, container } = renderPanel({
      camera: { live: true, onToggle: vi.fn(), videoRef: noopRef },
    });
    fireEvent.drop(container.querySelector("video")!.parentElement!, {
      dataTransfer: { files: [new File(["x"], "a.png")] },
    });
    expect(onFile).not.toHaveBeenCalled();
  });

  it("reports a camera failure in the RUN slot", () => {
    renderPanel({
      camera: {
        live: true,
        onToggle: vi.fn(),
        videoRef: noopRef,
        error: "Permission denied",
      },
    });
    expect(screen.getByRole("alert")).toHaveTextContent(/permission denied/i);
  });

  it("disables the still inputs while a decode or a run is in flight", () => {
    renderPanel({ busy: true, preparing: "file" });
    expect(screen.getByRole("button", { name: /decoding/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^Tiger$/ })).toBeDisabled();
  });
});
