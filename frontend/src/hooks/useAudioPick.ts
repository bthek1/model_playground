// Picking audio, for every audio route — the counterpart of `useImagePick`.
//
// Before this existed, the four audio routes had no input stage at all: `onFile`
// decoded a file *and* ran the model, `Record 5s` captured *and* ran, and a
// sample button fetched *and* ran. That collapsed INPUT and RUN into one
// gesture, with three consequences a user feels:
//
//   * there was no way to re-run the clip you already had — you re-uploaded it;
//   * changing a parameter (CLAP's labels, the VAD threshold's companion knobs)
//     meant capturing the audio again;
//   * browsing the samples spent one inference per click.
//
// So this hook holds a **decoded clip and nothing else**. Getting audio in and
// running a model on it are separate now, and the RUN button is the only thing
// that does the second (model-page-pattern.md §1.6).
//
// The one non-obvious rule it owns: **the clip is handed out as a copy.** Every
// audio worker takes the `Float32Array`'s buffer as a transfer, which detaches
// it on this side — so a page that ran the model on its own stored clip would
// find the waveform blank and the next run impossible. `take()` is the accessor
// that gets this right.

import { useCallback, useState } from "react";

import { decodeToMono, recordMic } from "@/audio/io";
import type { AudioSample } from "@/audio/samples";

export interface PickedAudio {
  /** Mono PCM at `sampleRate`. Never handed to a worker directly — see `take`. */
  audio: Float32Array;
  sampleRate: number;
  /** File name, sample label, or "Recording". */
  name: string;
  /** The bundled sample this came from, for a route that shows its transcript. */
  sample: AudioSample | null;
}

export type AudioPickKind = "file" | "mic" | "sample";

export interface UseAudioPickResult {
  clip: PickedAudio | null;
  /** Which source is being prepared, for the button that started it. */
  preparing: AudioPickKind | null;
  /** A failed decode, fetch, or denied mic. Belongs to the RUN slot. */
  error: string | null;
  clearError: () => void;
  pickFile: (file: File | undefined) => void;
  pickSample: (sample: AudioSample) => void;
  record: (seconds: number) => void;
  clear: () => void;
  /**
   * A detachable copy of the held clip, for handing to a worker. Returns null
   * when nothing is picked, so a caller can `const audio = take(); if (!audio)
   * return;` without reaching into `clip`.
   */
  take: () => Float32Array | null;
}

export function useAudioPick({
  /** Routes that aren't 16 kHz say so — `/audio-to-audio` is 48 kHz. */
  sampleRate = 16_000,
}: { sampleRate?: number } = {}): UseAudioPickResult {
  const [clip, setClip] = useState<PickedAudio | null>(null);
  const [preparing, setPreparing] = useState<AudioPickKind | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pick = useCallback(
    (
      open: () => Promise<{ audio: Float32Array; name: string; sample?: AudioSample }>,
      kind: AudioPickKind,
    ) => {
      setError(null);
      setPreparing(kind);
      void (async () => {
        try {
          const next = await open();
          setClip({
            audio: next.audio,
            sampleRate,
            name: next.name,
            sample: next.sample ?? null,
          });
        } catch (e) {
          setError(e instanceof Error ? e.message : String(e));
        } finally {
          setPreparing(null);
        }
      })();
    },
    [sampleRate],
  );

  const pickFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      pick(
        async () => ({
          audio: await decodeToMono(await file.arrayBuffer(), sampleRate),
          name: file.name,
        }),
        "file",
      );
    },
    [pick, sampleRate],
  );

  const pickSample = useCallback(
    (sample: AudioSample) =>
      pick(async () => {
        const response = await fetch(sample.url);
        if (!response.ok) {
          throw new Error(`Couldn't fetch ${sample.label} (HTTP ${response.status})`);
        }
        return {
          audio: await decodeToMono(await response.arrayBuffer(), sampleRate),
          name: sample.label,
          sample,
        };
      }, "sample"),
    [pick, sampleRate],
  );

  const record = useCallback(
    (seconds: number) =>
      pick(
        async () => ({
          audio: await recordMic(seconds, sampleRate),
          name: `Recording · ${seconds}s`,
        }),
        "mic",
      ),
    [pick, sampleRate],
  );

  const clear = useCallback(() => setClip(null), []);
  const clearError = useCallback(() => setError(null), []);
  // `slice()` and not the array itself: the worker detaches what it is given.
  const take = useCallback(() => clip?.audio.slice() ?? null, [clip]);

  return {
    clip,
    preparing,
    error,
    clearError,
    pickFile,
    pickSample,
    record,
    clear,
    take,
  };
}
