import { describe, expect, it, vi } from "vitest";

import {
  fitWithin,
  fromPayload,
  fromVideo,
  openCamera,
  toPayload,
  transferablesOf,
  type ImagePayload,
} from "./image";

/** A `RawImage`-shaped stand-in — the helpers only touch these four fields. */
function raw(width = 4, height = 3, fill = 7) {
  const data = new Uint8ClampedArray(width * height * 3);
  data.fill(fill);
  return { data, width, height, channels: 3 as const };
}

describe("fitWithin", () => {
  it("leaves an image that already fits untouched", () => {
    expect(fitWithin(640, 480, 640)).toEqual({ width: 640, height: 480 });
    expect(fitWithin(100, 100, 640)).toEqual({ width: 100, height: 100 });
  });

  it("caps the longest side and preserves the aspect ratio", () => {
    expect(fitWithin(1280, 720, 640)).toEqual({ width: 640, height: 360 });
    expect(fitWithin(720, 1280, 640)).toEqual({ width: 360, height: 640 });
  });

  it("never rounds a dimension down to zero", () => {
    expect(fitWithin(4000, 3, 640)).toEqual({ width: 640, height: 1 });
  });
});

describe("toPayload / fromPayload", () => {
  it("round-trips the pixels and the three numbers that describe them", () => {
    const source = raw();
    const payload = toPayload(source as never);
    const rebuilt = fromPayload(payload);

    expect(rebuilt.width).toBe(4);
    expect(rebuilt.height).toBe(3);
    expect(rebuilt.channels).toBe(3);
    expect(Array.from(rebuilt.data)).toEqual(Array.from(source.data));
  });

  it("copies by default, so the page's own image survives being sent", () => {
    const source = raw();
    const payload = toPayload(source as never);
    expect(payload.data).not.toBe(source.data);
    expect(payload.data.buffer).not.toBe(source.data.buffer);
  });

  it("shares the buffer when the caller says the frame is spent", () => {
    const source = raw();
    const payload = toPayload(source as never, { copy: false });
    expect(payload.data).toBe(source.data);
  });

  it("offers the pixel buffer as the transfer list", () => {
    const payload: ImagePayload = toPayload(raw() as never);
    expect(transferablesOf(payload)).toEqual([payload.data.buffer]);
  });
});

describe("fromVideo", () => {
  it("captures the frame at the video's own dimensions", () => {
    const drawImage = vi.fn();
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ({
        drawImage,
        // RawImage.fromCanvas reads the pixels back out; happy-dom's canvas has
        // no 2D implementation, so the context is a stand-in for both calls.
        getImageData: (_x: number, _y: number, w: number, h: number) => ({
          data: new Uint8ClampedArray(w * h * 4),
          width: w,
          height: h,
        }),
      }),
    } as unknown as HTMLCanvasElement;
    vi.spyOn(document, "createElement").mockReturnValueOnce(canvas);

    const video = { videoWidth: 320, videoHeight: 240 } as HTMLVideoElement;
    fromVideo(video);

    expect(canvas.width).toBe(320);
    expect(canvas.height).toBe(240);
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0);
  });
});

describe("openCamera", () => {
  it("stops every track when the returned stopper is called", async () => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
    const stream = { getTracks: () => tracks } as unknown as MediaStream;
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
      configurable: true,
    });
    const video = { play: vi.fn().mockResolvedValue(undefined), srcObject: null } as unknown as HTMLVideoElement;

    const stop = await openCamera(video, { width: 320, height: 240 });
    expect(video.srcObject).toBe(stream);

    stop();
    // A leaked MediaStream leaves the webcam light on after the user has
    // navigated away — the one bug in this module a user sees from across a room.
    expect(tracks[0].stop).toHaveBeenCalledOnce();
    expect(tracks[1].stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
  });
});
