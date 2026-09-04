// The ASR route's two bulky input surfaces, lifted out of the route so it can be
// layout + task glue like every other model page
// (model-page-restructure.md Phase 4).
//
// Both belong to the RUN slot, not OUTPUT: a sample clip and a retained take are
// *inputs* you can re-run a model over. Only the transcript is output.

import {
  AudioLines,
  Download,
  FlaskConical,
  Loader2,
  Pause,
  Play,
  Square,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { play, toWavBlob } from "@/audio/io";
import type { AudioSample } from "@/audio/samples";
import { AUDIO_SAMPLES } from "@/audio/samples";
import { formatDuration } from "@/audio/waveform";
import { LiveWaveform, Waveform } from "@/components/audio/Waveform";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Known-good clips with reference transcripts — an end-to-end model health check
 * that doesn't depend on the user's mic. Garbage output here means the model,
 * not the microphone.
 */
export function SampleClips({
  selected,
  loadingId,
  disabled,
  onRun,
}: {
  selected: AudioSample | null;
  /** Id of the clip currently being fetched/decoded, if any. */
  loadingId: string | null;
  disabled: boolean;
  onRun: (sample: AudioSample) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="size-4" /> Test with a sample clip
        </CardTitle>
        <CardDescription>
          Not sure the model works? Run a known clip and compare the output below
          with the reference. Garbage here means the model, not your mic.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {AUDIO_SAMPLES.map((s) => (
            <Button
              key={s.id}
              variant={selected?.id === s.id ? "default" : "outline"}
              size="sm"
              disabled={disabled}
              onClick={() => onRun(s)}
              title={s.hint}
            >
              {loadingId === s.id ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FlaskConical className="size-4" />
              )}
              {s.label}
            </Button>
          ))}
        </div>
        {selected && (
          <div className="rounded-md border bg-muted/30 px-3 py-2 text-sm">
            <p className="text-xs text-muted-foreground">{selected.hint}</p>
            {selected.reference ? (
              <p className="mt-1">
                <span className="font-medium text-muted-foreground">
                  Reference:{" "}
                </span>
                {selected.reference}
              </p>
            ) : (
              <p className="mt-1 text-muted-foreground">
                Long-form clip — no fixed reference; check that the transcript
                reads as coherent English with sensible timestamps.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * The live mic signal while recording, and the retained take afterwards —
 * replayable, downloadable, and re-runnable through another model.
 *
 * Playback state lives here rather than in the route: it is entirely derived
 * from `clip`, and nothing else on the page reads it.
 */
export function AudioTake({
  recording,
  stream,
  clip,
  sampleRate,
  canTranscribe,
  transcribing,
  onTranscribe,
}: {
  recording: boolean;
  stream: MediaStream | null;
  clip: Float32Array | null;
  sampleRate: number;
  canTranscribe: boolean;
  transcribing: boolean;
  onTranscribe: (clip: Float32Array) => void;
}) {
  const [playState, setPlayState] = useState<"idle" | "playing" | "paused">(
    "idle",
  );
  const playbackRef = useRef<AudioContext | null>(null);

  // A new take (or unmount) invalidates any in-progress playback.
  useEffect(() => {
    setPlayState("idle");
    const ref = playbackRef;
    return () => {
      const ctx = ref.current;
      ref.current = null;
      if (ctx && ctx.state !== "closed") void ctx.close();
    };
  }, [clip]);

  if (!recording && !clip) return null;

  // Play ⇄ Pause/Resume via the AudioContext transport. A fresh play() starts
  // from the top; suspend()/resume() pause/resume in place; the clip's natural
  // end resets to idle.
  const onPlayPause = () => {
    if (!clip) return;
    const ctx = playbackRef.current;
    if (playState === "playing" && ctx) {
      void ctx.suspend();
      setPlayState("paused");
    } else if (playState === "paused" && ctx) {
      void ctx.resume();
      setPlayState("playing");
    } else {
      const started = play(clip, sampleRate, () => {
        if (playbackRef.current === started) {
          playbackRef.current = null;
          setPlayState("idle");
        }
      });
      playbackRef.current = started;
      setPlayState("playing");
    }
  };

  const stopPlayback = () => {
    const ctx = playbackRef.current;
    playbackRef.current = null;
    setPlayState("idle");
    if (ctx && ctx.state !== "closed") void ctx.close();
  };

  const onDownload = () => {
    if (!clip) return;
    const url = URL.createObjectURL(toWavBlob(clip, sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = "recording.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <AudioLines className="size-4" /> Audio
          {clip && !recording && (
            <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">
              {formatDuration(clip.length, sampleRate)} · 16 kHz mono
            </span>
          )}
        </CardTitle>
        <CardDescription>
          {recording
            ? "Live microphone signal"
            : "Your take — replay it, download it, or run another model on it"}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {recording && stream ? (
          <LiveWaveform stream={stream} />
        ) : (
          clip && (
            <>
              <Waveform samples={clip} />
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" onClick={onPlayPause}>
                  {playState === "playing" ? (
                    <>
                      <Pause className="size-4" /> Pause
                    </>
                  ) : (
                    <>
                      <Play className="size-4" />{" "}
                      {playState === "paused" ? "Resume" : "Play"}
                    </>
                  )}
                </Button>
                {playState !== "idle" && (
                  <Button variant="outline" size="sm" onClick={stopPlayback}>
                    <Square className="size-4" /> Stop
                  </Button>
                )}
                <Button variant="outline" size="sm" onClick={onDownload}>
                  <Download className="size-4" /> Download WAV
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!canTranscribe}
                  onClick={() => onTranscribe(clip)}
                >
                  {transcribing ? (
                    <>
                      <Loader2 className="size-4 animate-spin" /> Transcribing…
                    </>
                  ) : (
                    <>
                      <AudioLines className="size-4" /> Transcribe clip
                    </>
                  )}
                </Button>
              </div>
            </>
          )
        )}
      </CardContent>
    </Card>
  );
}
