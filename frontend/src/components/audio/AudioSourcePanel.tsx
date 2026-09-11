// The input surface every audio route puts in its RUN slot — the counterpart of
// `ImageSourcePanel`, and for the same reason: a record button, an upload
// button, the bundled clips and a waveform of whatever is currently held.
//
// It renders the **input** only. The enhanced audio, the ranked tags, the
// speech segments, the transcript — those are results, and results live in
// OUTPUT (model-page-pattern.md §4).
//
// Getting audio in is not running a model. Each of these sources decodes to a
// `Float32Array` and stops there; the waveform below is the receipt, and the
// route's RUN button is what spends a model on it. These buttons are therefore
// **not gated on `ready`** — picking a clip before downloading weights is a
// perfectly sensible order to work in, and the old gate forced the user to load
// a model before they were allowed to choose what to run it on.

import { Loader2, Mic, Upload } from "lucide-react";
import { useRef, type ReactNode } from "react";

import { Waveform } from "@/components/audio/Waveform";
import { Button } from "@/components/ui/button";
import type { AudioSample } from "@/audio/samples";
import type { AudioPickKind, PickedAudio } from "@/hooks/useAudioPick";
import { cn } from "@/lib/utils";

export function AudioSourcePanel({
  clip,
  preparing,
  samples,
  sampleHint,
  onFile,
  onSample,
  onRecord,
  recordSeconds,
  busy,
  children,
}: {
  clip: PickedAudio | null;
  preparing: AudioPickKind | null;
  /** Omit on a route with no bundled clips. */
  samples?: readonly AudioSample[];
  /** One line above the sample row: what these particular clips are for. */
  sampleHint?: ReactNode;
  onFile: (file: File | undefined) => void;
  onSample?: (sample: AudioSample) => void;
  /** Omit to hide the mic button — a route with no capture path. */
  onRecord?: (seconds: number) => void;
  recordSeconds?: number;
  /** A run in flight. Disables the sources so the held clip can't change under it. */
  busy: boolean;
  /** Extra task controls — a label editor, a threshold slider. */
  children?: ReactNode;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const seconds = recordSeconds ?? 5;

  return (
    // Scrolls its own overflow, for the same reason `ImageSourcePanel` does:
    // `InputPanel` puts the transport row below this as a `sticky bottom-0`
    // sibling, so a tall input surface has to scroll here rather than spill
    // over that bar. `md:`, so a phone is never trapped in a nested scroller.
    <div className="flex min-h-0 flex-1 flex-col gap-3 md:overflow-y-auto">
      <div className="flex flex-wrap items-center gap-2">
        {onRecord && (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => onRecord(seconds)}
          >
            {preparing === "mic" ? (
              <>
                <Loader2 className="size-4 animate-spin" /> Recording…
              </>
            ) : (
              <>
                <Mic className="size-4" /> Record {seconds}s
              </>
            )}
          </Button>
        )}
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
        >
          {preparing === "file" ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Decoding…
            </>
          ) : (
            <>
              <Upload className="size-4" /> Upload audio
            </>
          )}
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = ""; // allow re-selecting the same file
            onFile(file);
          }}
        />
      </div>

      {samples && samples.length > 0 && onSample && (
        <div className="space-y-1.5">
          {sampleHint && (
            <p className="text-xs text-muted-foreground">{sampleHint}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {samples.map((sample) => (
              <Button
                key={sample.id}
                size="sm"
                variant={clip?.sample?.id === sample.id ? "secondary" : "ghost"}
                className="h-7 border text-xs"
                disabled={busy}
                title={sample.hint}
                onClick={() => onSample(sample)}
              >
                {preparing === "sample" && clip?.sample?.id !== sample.id ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : null}
                {sample.label}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* The held clip. This is the whole point of separating INPUT from RUN:
          there is now a "what am I about to run on?" state on screen, and it
          survives across runs and parameter changes. */}
      <div
        data-testid="audio-input"
        className={cn(
          "rounded-md border p-3",
          clip ? "bg-muted/30" : "bg-muted/10",
        )}
      >
        {clip ? (
          <div className="space-y-2">
            <p className="flex flex-wrap items-baseline gap-x-2 text-xs text-muted-foreground">
              <span className="truncate font-medium text-foreground">
                {clip.name}
              </span>
              <span className="tabular-nums">
                {(clip.audio.length / clip.sampleRate).toFixed(1)}s ·{" "}
                {(clip.sampleRate / 1000).toFixed(clip.sampleRate % 1000 ? 1 : 0)} kHz
              </span>
            </p>
            <Waveform samples={clip.audio} className="text-muted-foreground" />
          </div>
        ) : (
          <p
            data-testid="audio-input-empty"
            className="py-4 text-center text-sm text-muted-foreground"
          >
            No audio yet — record, upload, or pick a clip above. Nothing runs
            until you press the button below.
          </p>
        )}
      </div>

      {children}
    </div>
  );
}
