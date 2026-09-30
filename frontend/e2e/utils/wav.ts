// A WAV for Chromium's fake microphone.
//
// `--use-file-for-fake-audio-capture=<path>%noloop` plays a 16-bit PCM file as
// the mic and then goes silent, which is the only way to put a *known* take in
// front of a live capture loop. The file has to exist before the browser
// launches, so it is built in Node from a clip fetched once.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const FAKE_MIC_RATE = 16000;

/** 16-bit PCM WAV → mono Float32 at {@link FAKE_MIC_RATE} (box low-pass + linear resample). */
export function decodeWav(buf: Buffer): Float32Array {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let at = 12;
  let channels = 1;
  let rate = FAKE_MIC_RATE;
  let bits = 16;
  while (at + 8 <= buf.length) {
    const id = buf.toString("ascii", at, at + 4);
    const size = dv.getUint32(at + 4, true);
    if (id === "fmt ") {
      channels = dv.getUint16(at + 10, true);
      rate = dv.getUint32(at + 12, true);
      bits = dv.getUint16(at + 22, true);
    } else if (id === "data") {
      if (bits !== 16) throw new Error(`expected 16-bit PCM, got ${bits}`);
      const n = size / 2 / channels;
      const mono = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let v = 0;
        for (let c = 0; c < channels; c++) v += dv.getInt16(at + 8 + (i * channels + c) * 2, true);
        mono[i] = v / channels / 32768;
      }
      if (rate === FAKE_MIC_RATE) return mono;
      const ratio = rate / FAKE_MIC_RATE;
      const k = Math.max(1, Math.round(ratio));
      const out = new Float32Array(Math.floor(n / ratio));
      for (let i = 0; i < out.length; i++) {
        const x = i * ratio;
        const j = Math.floor(x);
        let a = 0;
        let b = 0;
        for (let d = 0; d < k; d++) {
          a += mono[Math.min(n - 1, j + d)];
          b += mono[Math.min(n - 1, j + 1 + d)];
        }
        out[i] = ((1 - (x - j)) * a + (x - j) * b) / k;
      }
      return out;
    }
    at += 8 + size + (size & 1);
  }
  throw new Error("no data chunk");
}

/** Mono Float32 → 16-bit PCM WAV at {@link FAKE_MIC_RATE}. */
export function encodeWav(samples: Float32Array): Buffer {
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + samples.length * 2, 4);
  buf.write("WAVEfmt ", 8, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(FAKE_MIC_RATE, 24);
  buf.writeUInt32LE(FAKE_MIC_RATE * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  return buf;
}

const JFK_URL =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";

/**
 * `leadSeconds` of silence, JFK's 11 s, then `tailSeconds` of silence — written
 * to a fixed temp path (cached across runs) and returned.
 */
export async function silenceSpeechSilenceWav(
  leadSeconds: number,
  tailSeconds: number,
): Promise<string> {
  const dir = join(tmpdir(), "model-playground-e2e");
  await mkdir(dir, { recursive: true });
  const path = join(dir, `fake-mic-jfk-${leadSeconds}-${tailSeconds}.wav`);
  if (existsSync(path)) return path;

  const source = join(dir, "jfk.wav");
  if (!existsSync(source)) {
    const res = await fetch(JFK_URL);
    if (!res.ok) throw new Error(`${JFK_URL}: HTTP ${res.status}`);
    await writeFile(source, Buffer.from(await res.arrayBuffer()));
  }
  const jfk = decodeWav(await readFile(source));
  const take = new Float32Array(
    Math.round((leadSeconds + tailSeconds) * FAKE_MIC_RATE) + jfk.length,
  );
  take.set(jfk, Math.round(leadSeconds * FAKE_MIC_RATE));
  await writeFile(path, encodeWav(take));
  return path;
}
