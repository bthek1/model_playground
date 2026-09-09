// The input surface every vision route puts in its RUN slot: a preview, a drop
// target, an upload button, the bundled samples, and — for the routes that run
// live — a camera toggle that swaps the still preview for a playing `<video>`.
//
// It renders the input only. The overlay (boxes, a depth map, class masks) is a
// *result*, and results live in OUTPUT (model-page-pattern.md §4). Painting the
// answer on top of the input would make the two slots one, which is precisely
// the drift the four-slot shell exists to prevent.

import { Camera, CameraOff, Loader2, Upload } from "lucide-react";
import { useRef, type ReactNode, type Ref } from "react";

import { Button } from "@/components/ui/button";
import type { PickedImage } from "@/hooks/useImagePick";
import type { ImageSample } from "@/vision/samples";

export interface CameraControl {
  /** True while the camera is the active source. */
  live: boolean;
  onToggle: (live: boolean) => void;
  /** Attached to the `<video>` the frame pump samples. A callback ref, usually. */
  videoRef: Ref<HTMLVideoElement>;
  /** A denied permission or a missing device. */
  error?: string | null;
  /** Measured end-to-end frames per second, once the loop has settled. */
  fps?: number | null;
}

export function ImageSourcePanel({
  picked,
  preparing,
  samples,
  sampleHint,
  onFile,
  onSample,
  busy,
  camera,
  children,
}: {
  picked: PickedImage | null;
  preparing: null | "file" | "sample";
  samples: readonly ImageSample[];
  /** One line above the sample row: what these particular pictures are for. */
  sampleHint: ReactNode;
  onFile: (file: File | undefined) => void;
  onSample: (sample: ImageSample) => void;
  busy: boolean;
  /** Omit on a route with no live mode. */
  camera?: CameraControl;
  /** Extra task controls — a threshold slider, a label editor. */
  children?: ReactNode;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const live = camera?.live ?? false;

  return (
    // **This surface scrolls its own overflow, and must.** `InputPanel` puts the
    // transport row below it as a `sticky bottom-0` sibling, and the RUN slot is
    // height-clamped from `md` up. Without `overflow-y-auto` a tall input — the
    // zero-shot page, with a label editor and a template editor above the sample
    // row — is squeezed by `flex-1` but spills its children *past* its own box,
    // painting them over the transport bar. The samples then look present and
    // are not clickable. Scrolling here keeps the bar genuinely below the
    // content instead of on top of it.
    //
    // `md:`, matching `ModelPage`'s work columns: below that breakpoint there is
    // no height clamp, so a phone is never trapped in a nested scroll container.
    <div className="flex min-h-0 flex-1 flex-col gap-3 md:overflow-y-auto">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!live) onFile(e.dataTransfer.files?.[0]);
        }}
        className="flex min-h-40 flex-1 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-dashed bg-muted/20 p-2"
      >
        {/* The <video> is mounted whenever the route has a camera, not only
            while it is live: `useCamera` needs the element to exist before it
            can attach a stream to it, and a ref to a node React has not created
            yet is null at exactly the wrong moment. */}
        {camera && (
          <video
            ref={camera.videoRef}
            muted
            playsInline
            aria-label="Camera preview"
            className={live ? "max-h-72 max-w-full rounded" : "hidden"}
          />
        )}
        {!live &&
          (picked ? (
            <img
              src={picked.previewUrl}
              alt={`Selected input: ${picked.name}`}
              className="max-h-72 max-w-full rounded object-contain"
            />
          ) : (
            <p className="px-4 text-center text-sm text-balance text-muted-foreground">
              Drop an image here, upload one, or start from a sample below.
            </p>
          ))}
      </div>

      {camera?.error && (
        <p role="alert" className="text-xs text-destructive">
          {camera.error}
        </p>
      )}

      {children}

      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">{sampleHint}</p>
        <div className="flex flex-wrap gap-2">
          {samples.map((sample) => (
            <Button
              key={sample.id}
              variant="outline"
              size="sm"
              disabled={busy || live}
              title={sample.hint}
              onClick={() => onSample(sample)}
            >
              {sample.label}
            </Button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={busy || live}
          onClick={() => fileRef.current?.click()}
        >
          {preparing === "file" ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Decoding…
            </>
          ) : (
            <>
              <Upload className="size-4" /> Upload image
            </>
          )}
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          aria-label="Upload an image"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = ""; // allow re-selecting the same file
            onFile(file);
          }}
        />

        {camera && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => camera.onToggle(!live)}
          >
            {live ? (
              <>
                <CameraOff className="size-4" /> Stop camera
              </>
            ) : (
              <>
                <Camera className="size-4" /> Use camera
              </>
            )}
          </Button>
        )}

        {live && camera?.fps != null && (
          <span
            data-testid="live-fps"
            className="self-center font-mono text-xs text-muted-foreground tabular-nums"
          >
            {camera.fps} fps
          </span>
        )}
      </div>
    </div>
  );
}
