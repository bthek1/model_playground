// Browser audio I/O — the client-side equivalent of the notebooks' `soundfile` +
// `librosa`. Decoding/resampling uses the Web Audio API; ASR/classification models
// want 16 kHz mono Float32, which `decodeToMono` produces for any input format.

/**
 * Decode any audio `ArrayBuffer` (wav/mp3/ogg/flac/…) to mono `Float32` at a
 * target sample rate. An `OfflineAudioContext` created at `targetRate` resamples
 * for free when the decoded buffer is rendered through it.
 */
export async function decodeToMono(
  data: ArrayBuffer,
  targetRate = 16000,
): Promise<Float32Array> {
  // `decodeAudioData` may detach the buffer, so decode a copy.
  const tmp = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await tmp.decodeAudioData(data.slice(0));
  } finally {
    await tmp.close();
  }

  const frames = Math.max(1, Math.ceil(decoded.duration * targetRate));
  const off = new OfflineAudioContext(1, frames, targetRate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0); // mono Float32 @ targetRate
}

/**
 * Capture `seconds` of audio from the microphone as mono `Float32` at
 * `targetRate`. Prompts for mic permission; stops the tracks when done.
 */
export async function recordMic(
  seconds: number,
  targetRate = 16000,
): Promise<Float32Array> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    const rec = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    const stopped = new Promise<void>((resolve) => {
      rec.onstop = () => resolve();
    });
    rec.start();
    await new Promise((r) => setTimeout(r, seconds * 1000));
    rec.stop();
    await stopped;
    const buf = await new Blob(chunks).arrayBuffer();
    return decodeToMono(buf, targetRate);
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
}

/**
 * Play mono `Float32` samples through the speakers. Returns the `AudioContext`
 * so the caller can `suspend()` (pause) / `resume()` / `close()` (stop) it.
 * `onEnded` fires when the clip finishes on its own (not on `close()`).
 */
export function play(
  samples: Float32Array,
  sampleRate: number,
  onEnded?: () => void,
): AudioContext {
  const ctx = new AudioContext();
  const buf = ctx.createBuffer(1, samples.length, sampleRate);
  buf.getChannelData(0).set(samples);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  if (onEnded) src.onended = () => onEnded();
  src.start();
  return ctx;
}

/**
 * Encode mono `Float32` samples as a 16-bit PCM WAV `Blob` (downloadable) — the
 * browser analog of the notebooks' `save_wav`.
 */
export function toWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // audio format = PCM
  view.setUint16(22, 1, true); // channels = mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}
