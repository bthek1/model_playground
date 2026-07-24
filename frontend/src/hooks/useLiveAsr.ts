import { useCallback, useEffect, useRef, useState } from "react";

import { decodeToMono } from "@/audio/io";
import { DEFAULT_ASR_MODEL, type AsrRunArgs } from "@/audio/types";
import { useAsr } from "@/hooks/useAsr";

const TARGET_RATE = 16000;
/** Re-transcribe at most the last N seconds — Whisper's native chunk size. */
const WINDOW_SECONDS = 30;
/** How often MediaRecorder emits a chunk (and we attempt a re-transcription). */
const TIMESLICE_MS = 1500;

export interface UseLiveAsrResult {
  /** Model-load status, forwarded from `useAsr`. */
  status: ReturnType<typeof useAsr>["status"];
  ready: boolean;
  loading: boolean;
  progress: ReturnType<typeof useAsr>["progress"];
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
  error: string | null;
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
}

/**
 * Real-time speech-to-text. Captures the mic continuously with `MediaRecorder`
 * and, on each emitted chunk, decodes the take so far and re-transcribes its last
 * {@link WINDOW_SECONDS} through the ASR worker — so the transcript updates live
 * as the user speaks. Overlapping ticks are skipped (one transcription in flight
 * at a time); the growing audio keeps accumulating and the next free tick catches
 * up. The full take is retained as `clip` when capture stops, so the UI can
 * visualize, replay, download, or re-transcribe it. Built on {@link useAsr}, so
 * model loading/backends are shared.
 */
export function useLiveAsr(model: string = DEFAULT_ASR_MODEL): UseLiveAsrResult {
  const asr = useAsr(model);
  const { transcribe, ready } = asr;

  const [recording, setRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [clip, setClip] = useState<Float32Array | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const busyRef = useRef(false);

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
        const windowed = full.slice(Math.max(0, full.length - maxSamples));
        const res = await transcribe(windowed);
        setText(res.text);
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
  }, []);

  const start = useCallback(async () => {
    if (!ready) {
      setError("Model is still loading — try again in a moment.");
      return;
    }
    setError(null);
    setText("");
    setClip(null);
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
    ready: asr.ready,
    loading: asr.loading,
    progress: asr.progress,
    backend: asr.backend,
    recording,
    running: asr.running,
    stream,
    clip,
    sampleRate: TARGET_RATE,
    text,
    error: error ?? asr.error,
    start,
    stop,
    transcribeClip,
  };
}
