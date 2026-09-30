import { useCallback, useEffect, useRef, useState } from "react";

import { decodeToMono } from "@/audio/io";
import { sliceHasSpeech } from "@/audio/vad/liveGate";
import { FRAME_SAMPLES } from "@/audio/vad/vad";
import {
  DEFAULT_ASR_MODEL,
  type AsrChunk,
  type AsrRunArgs,
} from "@/audio/types";
import { useAsr } from "@/hooks/useAsr";
import type { LoadProgress } from "@/model/progress";

const TARGET_RATE = 16000;
/** Re-transcribe at most the last N seconds — Whisper's native chunk size. */
const WINDOW_SECONDS = 30;
/** How often MediaRecorder emits a chunk (and we attempt a re-transcription). */
const TIMESLICE_MS = 1500;

/**
 * Shift segment timestamps from window-relative to take-relative. The live loop
 * only re-transcribes the tail {@link WINDOW_SECONDS}, so once a take is longer
 * than that the model's timestamps restart at 0 and would otherwise rewind on
 * screen mid-recording.
 */
function shiftChunks(chunks: AsrChunk[] | undefined, offset: number): AsrChunk[] {
  if (!chunks?.length) return [];
  if (!offset) return chunks;
  return chunks.map((c) => ({
    ...c,
    timestamp: [
      c.timestamp[0] + offset,
      c.timestamp[1] == null ? null : c.timestamp[1] + offset,
    ] as [number, number | null],
  }));
}

export interface UseLiveAsrResult {
  /** Model-load status, forwarded from `useAsr`. */
  status: ReturnType<typeof useAsr>["status"];
  /** True before the user has asked for the weights. */
  idle: boolean;
  ready: boolean;
  loading: boolean;
  progress: ReturnType<typeof useAsr>["progress"];
  /** Aggregate load progress across every file. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** Duration of the load that produced `ready`, in ms. */
  loadedInMs: number | null;
  backend: string | null;
  /** True while the mic is live and transcription is updating. */
  recording: boolean;
  /** True while a transcription request is in flight (mic tick or clip). */
  running: boolean;
  /** The live mic stream while recording (for waveform visualization). */
  stream: MediaStream | null;
  /**
   * The retained audio (mono Float32 @ {@link sampleRate}): the full take after
   * `stop()`, or the uploaded clip after `transcribeClip`. Null until then.
   */
  clip: Float32Array | null;
  /** Sample rate of `clip` (and of everything sent to the model). */
  sampleRate: number;
  /** The latest transcript text (grows/refines live while recording). */
  text: string;
  /**
   * The transcript split into timestamped segments, when the model returns them.
   * Timestamps are relative to the **whole take**, not the re-transcribed window
   * (see `runWindow`). Empty when the model returned no segmentation.
   */
  chunks: AsrChunk[];
  error: string | null;
  /**
   * Whether live ticks with no new speech skip the model (default on). Applies
   * to the live loop only — never to the final pass on `stop()`, and never to
   * `transcribeClip`. Flipping it runs nothing; the next tick reads it.
   */
  skipSilence: boolean;
  setSkipSilence: (on: boolean) => void;
  /**
   * True while the latest live tick was skipped for silence — the transcript is
   * standing still because nothing was said, not because the model hung.
   * Cleared by the next transcribed tick, by `stop()` and by `start()`.
   */
  silent: boolean;
  /** Live ticks skipped for silence in this take. */
  skippedTicks: number;
  /** Begin live capture + transcription. Requests mic permission. */
  start: () => Promise<void>;
  /** Stop capture; runs one final transcription over the full take. */
  stop: () => void;
  /**
   * One-shot transcription of a ready Float32 clip (e.g. an uploaded file, or
   * the retained `clip` to re-apply a newly selected model). The clip is kept
   * for visualization/playback — a copy is sent to the worker, since transfer
   * would detach the caller's buffer.
   */
  transcribeClip: (audio: Float32Array, args?: AsrRunArgs) => Promise<void>;
  /** Start the weight download (no-op unless idle). */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: () => void;
  /** Abandon a load in flight, returning to `idle`. */
  cancel: () => void;
}

/**
 * Real-time speech-to-text. Captures the mic continuously with `MediaRecorder`
 * and, on each emitted chunk, decodes the take so far and re-transcribes its last
 * {@link WINDOW_SECONDS} through the ASR worker — so the transcript updates live
 * as the user speaks. Overlapping ticks are skipped (one transcription in flight
 * at a time); the growing audio keeps accumulating and the next free tick catches
 * up. Ticks whose new audio holds no speech skip the model too
 * (`audio/vad/liveGate.ts`): Whisper transcribes room tone as "you", and each
 * skipped pass is ~0.75 s of inference saved. The full take is retained as `clip` when capture stops, so the UI can
 * visualize, replay, download, or re-transcribe it. Built on {@link useAsr}, so
 * model loading/backends are shared.
 */
export function useLiveAsr(
  model: string = DEFAULT_ASR_MODEL,
  autoLoad = false,
): UseLiveAsrResult {
  const asr = useAsr(model, autoLoad);
  const { transcribe, ready } = asr;

  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [clip, setClip] = useState<Float32Array | null>(null);
  const [text, setText] = useState("");
  const [chunks, setChunks] = useState<AsrChunk[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [skipSilence, setSkipSilence] = useState(true);
  const [silent, setSilent] = useState(false);
  const [skippedTicks, setSkippedTicks] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const busyRef = useRef(false);
  // Read by the tick, not captured by it: a toggle mid-take applies to the next
  // tick rather than waiting for a new `runWindow` closure.
  const skipSilenceRef = useRef(true);
  const setSkip = useCallback((on: boolean) => {
    skipSilenceRef.current = on;
    setSkipSilence(on);
  }, []);
  // Samples of the take already judged by the gate. Advanced on every scored
  // tick — transcribed or skipped — but not on a tick dropped because another
  // was in flight, so the next free tick judges everything that arrived since.
  const scoredRef = useRef(0);

  // Decode the take-so-far and transcribe its tail window. On the final pass
  // (recorder stopped) the full take is retained as `clip`. Silent on transient
  // decode errors — early partial streams aren't always decodable mid-capture.
  const runWindow = useCallback(
    async (final = false) => {
      if (chunksRef.current.length === 0) return;
      if (busyRef.current && !final) return;
      busyRef.current = true;
      try {
        const blob = new Blob(chunksRef.current, { type: chunksRef.current[0].type });
        const full = await decodeToMono(await blob.arrayBuffer(), TARGET_RATE);
        if (final) setClip(full);
        // Always slice (= copy): `transcribe` transfers the buffer, and the
        // retained `full` must stay usable for visualization/playback.
        const maxSamples = WINDOW_SECONDS * TARGET_RATE;
        const start = Math.max(0, full.length - maxSamples);
        const windowed = full.slice(start);

        // The gate never touches the final pass: the last transcript must
        // reflect the whole take, silent tail or not.
        if (!final && skipSilenceRef.current) {
          // One frame of overlap, so a word whose onset straddles the tick
          // boundary is judged with its start.
          const from = Math.max(0, scoredRef.current - FRAME_SAMPLES) - start;
          scoredRef.current = full.length;
          let speech = true; // fail open: a gate that errors transcribes
          try {
            speech = sliceHasSpeech(windowed, from, TARGET_RATE).speech;
          } catch {
            /* keep speech = true */
          }
          if (!speech) {
            setSilent(true);
            setSkippedTicks((n) => n + 1);
            return;
          }
        }
        scoredRef.current = full.length;
        setSilent(false);

        const res = await transcribe(windowed);
        setText(res.text);
        setChunks(shiftChunks(res.chunks, start / TARGET_RATE));
      } catch {
        /* transient decode/transcribe failure mid-capture — keep listening */
      } finally {
        busyRef.current = false;
      }
    },
    [transcribe],
  );

  const stop = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop(); // fires onstop → final pass
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recorderRef.current = null;
    setStream(null);
    setRecording(false);
    setSilent(false);
  }, []);

  const start = useCallback(async () => {
    if (!ready) {
      setError("Model is still loading — try again in a moment.");
      return;
    }
    setError(null);
    setText("");
    setChunks([]);
    setClip(null);
    setSilent(false);
    setSkippedTicks(0);
    scoredRef.current = 0;
    try {
      const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = mic;
      setStream(mic);
      chunksRef.current = [];

      const rec = new MediaRecorder(mic);
      recorderRef.current = rec;
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
        void runWindow();
      };
      rec.onstop = () => void runWindow(true); // final pass — retains the take
      rec.start(TIMESLICE_MS);
      setRecording(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStream(null);
      setRecording(false);
    }
  }, [ready, runWindow]);

  const transcribeClip = useCallback(
    async (audio: Float32Array, args?: AsrRunArgs) => {
      setError(null);
      setClip(audio);
      try {
        // Send a copy — transfer would detach the retained clip's buffer.
        const res = await transcribe(audio.slice(), args);
        setText(res.text);
        setChunks(res.chunks ?? []);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [transcribe],
  );

  // Release the mic if the component unmounts mid-capture.
  useEffect(() => {
    return () => {
      const rec = recorderRef.current;
      if (rec && rec.state !== "inactive") rec.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    status: asr.status,
    idle: asr.idle,
    ready: asr.ready,
    loading: asr.loading,
    progress: asr.progress,
    loadProgress: asr.loadProgress,
    loadedInMs: asr.loadedInMs,
    backend: asr.backend,
    recording,
    running: asr.running,
    stream,
    clip,
    sampleRate: TARGET_RATE,
    text,
    chunks,
    error: error ?? asr.error,
    skipSilence,
    setSkipSilence: setSkip,
    silent,
    skippedTicks,
    start,
    stop,
    transcribeClip,
    load: asr.load,
    retry: asr.retry,
    cancel: asr.cancel,
  };
}
