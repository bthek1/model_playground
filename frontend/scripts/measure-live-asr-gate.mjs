// Phase 1 of #55: what does the live ASR loop do on silence, and which VAD
// should gate it?
//
//   node scripts/measure-live-asr-gate.mjs
//
// Builds synthetic takes out of jfk.wav — room noise, speech, a long pause,
// speech, a tail — and replays them as `useLiveAsr` sees them: a tick every
// 1.5 s, each one transcribing the last 30 s. At each tick it records
//
//   - what Whisper-base (the page's default) returns for the window,
//   - Silero v5's verdict on the new 1.5 s (≥ 3 frames above 0.5),
//   - the energy gate's verdict (`audio/vad/liveGate.ts`, imported, not copied).
//
// Downloads jfk.wav, silero (2 MB) and whisper-base q8 once, into the OS cache
// directory Transformers.js already uses. Runs on onnxruntime-node — where the
// quantized seq2seq decoder *does* open (the qdq bug is web-ORT only), which is
// fine for this question: it measures the model's behaviour, not the page's
// session.
//
// A synthetic take is not a real mic (no Opus round trip, no AGC), so this
// measures the decision the gate makes, not the room it will be used in.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { createJiti } from "jiti";
import * as ort from "onnxruntime-node";
import { pipeline } from "@huggingface/transformers";

const here = dirname(fileURLToPath(import.meta.url));
const jiti = createJiti(import.meta.url, { alias: { "@": join(here, "../src") } });
const { sliceHasSpeech } = await jiti.import("../src/audio/vad/liveGate.ts");
const { frameProbabilities, FRAME_SAMPLES } = await jiti.import("../src/audio/vad/vad.ts");

const SR = 16000;
const TICK = 1.5 * SR;
const WINDOW = 30 * SR;
const cache = join(tmpdir(), "model-playground-measure");
await mkdir(cache, { recursive: true });

async function cached(url, name) {
  const path = join(cache, name);
  if (!existsSync(path)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status}`);
    await writeFile(path, Buffer.from(await res.arrayBuffer()));
  }
  return readFile(path);
}

/**
 * 16-bit PCM WAV → mono Float32 @ 16 kHz. Downmixes, then resamples with a
 * box low-pass and linear interpolation — crude, but the question here is a
 * speech/no-speech verdict, not fidelity.
 */
function parseWav(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let at = 12;
  let fmt;
  while (at < buf.length) {
    const id = buf.toString("ascii", at, at + 4);
    const size = dv.getUint32(at + 4, true);
    if (id === "fmt ") {
      fmt = { channels: dv.getUint16(at + 10, true), rate: dv.getUint32(at + 12, true), bits: dv.getUint16(at + 22, true) };
    } else if (id === "data") {
      if (fmt.bits !== 16) throw new Error(`expected 16-bit PCM, got ${fmt.bits}`);
      const n = size / 2 / fmt.channels;
      const mono = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let v = 0;
        for (let c = 0; c < fmt.channels; c++) v += dv.getInt16(at + 8 + (i * fmt.channels + c) * 2, true);
        mono[i] = v / fmt.channels / 32768;
      }
      if (fmt.rate === SR) return mono;
      const ratio = fmt.rate / SR;
      const k = Math.max(1, Math.round(ratio));
      const out = new Float32Array(Math.floor(n / ratio));
      for (let i = 0; i < out.length; i++) {
        const x = i * ratio;
        const j = Math.floor(x);
        let a = 0, b = 0;
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

/** Deterministic noise at a target RMS level; `smooth` > 0 low-passes it (fan-like). */
function noise(seconds, db, smooth = 0, seed = 1) {
  const out = new Float32Array(Math.round(seconds * SR));
  let s = seed >>> 0;
  let y = 0;
  for (let i = 0; i < out.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const x = s / 2 ** 32 - 0.5;
    y = smooth * y + (1 - smooth) * x;
    out[i] = y;
  }
  let sum = 0;
  for (const v of out) sum += v * v;
  const gain = 10 ** (db / 20) / Math.sqrt(sum / out.length);
  for (let i = 0; i < out.length; i++) out[i] *= gain;
  return out;
}

function concat(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) out.set(p, (at += p.length) - p.length);
  return out;
}

function mix(a, b) {
  const out = a.slice();
  for (let i = 0; i < out.length; i++) out[i] += b[i % b.length];
  return out;
}

const jfk = parseWav(await cached(
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
  "jfk.wav",
));

const silero = await ort.InferenceSession.create(
  await cached("https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx", "silero.onnx"),
);
const infer = async (window, state) => {
  const out = await silero.run({
    input: new ort.Tensor("float32", window, [1, window.length]),
    state: new ort.Tensor("float32", state, [2, 1, 128]),
    sr: new ort.Tensor("int64", BigInt64Array.from([BigInt(SR)]), []),
  });
  return { probability: out.output.data[0], state: out.stateN.data };
};

const asr = await pipeline("automatic-speech-recognition", "onnx-community/whisper-base", {
  dtype: "q8",
  device: "cpu",
});

// Speech, a 12 s pause, speech again, then trailing silence — the take shape
// the gate exists for. Two rooms: a quiet one and a fan-level noise bed.
const silence = (s) => new Float32Array(Math.round(s * SR));
const clean = concat([silence(4.5), jfk, silence(12), jfk, silence(6)]);
// The third take is the gate's worst case: a talker who never pauses for longer
// than JFK does between sentences, for longer than the 30 s window, over the
// fan. The gate's floor is the window's quietest tenth, so a window with no
// real silence in it is where a floor could climb into the speech.
const talker = concat([jfk, jfk, jfk, jfk]);
const takes = [
  { name: "quiet room (-60 dBFS white)", audio: mix(clean, noise(clean.length / SR, -60, 0, 7)) },
  { name: "fan (-40 dBFS low-passed)", audio: mix(clean, noise(clean.length / SR, -40, 0.9, 11)) },
  { name: "no-pause talker over the fan", audio: mix(talker, noise(talker.length / SR, -40, 0.9, 13)) },
];
const only = process.argv[2];

for (const take of takes.filter((t) => !only || t.name.startsWith(only))) {
  const probs = await frameProbabilities(take.audio, infer);
  let prevText = "";
  const rows = [];
  for (let end = TICK, from = 0; end <= take.audio.length; from = end, end += TICK) {
    const start = Math.max(0, end - WINDOW);
    const window = take.audio.slice(start, end);

    const f0 = Math.floor(from / FRAME_SAMPLES);
    const f1 = Math.ceil(end / FRAME_SAMPLES);
    let sileroFrames = 0;
    for (let f = f0; f < f1; f++) if (probs[f] > 0.5) sileroFrames++;
    const sileroSpeech = sileroFrames >= 3;

    const t0 = performance.now();
    const gate = sliceHasSpeech(window, from - start, SR);
    const gateMs = performance.now() - t0;

    const t1 = performance.now();
    const { text } = await asr(window);
    const asrMs = performance.now() - t1;

    rows.push({
      t: (end / SR).toFixed(1),
      silero: sileroSpeech ? "speech" : "-",
      energy: gate.speech ? "speech" : "-",
      agree: sileroSpeech === gate.speech ? "" : "DISAGREE",
      changed: text.trim() !== prevText.trim() ? "changed" : "",
      gateMs: gateMs.toFixed(2),
      asrMs: asrMs.toFixed(0),
      text: text.trim().slice(0, 60),
    });
    prevText = text;
  }

  const silent = rows.filter((r) => r.silero === "-");
  console.log(`\n== ${take.name}: ${rows.length} ticks, ${silent.length} with no new speech (Silero)`);
  console.table(rows);
  console.log({
    energySkips: rows.filter((r) => r.energy === "-").length,
    sileroSkips: silent.length,
    disagreements: rows.filter((r) => r.agree).length,
    silentTicksWhoseTextChanged: silent.filter((r) => r.changed).length,
    silentTicksWithAnyText: silent.filter((r) => r.text).length,
    meanAsrMsOnSilentTicks: Math.round(silent.reduce((n, r) => n + Number(r.asrMs), 0) / Math.max(1, silent.length)),
  });
}
