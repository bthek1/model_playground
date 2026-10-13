import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Real-time voice-to-text, entirely in your browser — a Whisper/Moonshine ONNX model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker.",
  about: [
    "Automatic speech recognition, usually just called speech to text, turns spoken words into a written transcript. This page does it in the tab: press Start listening and the transcript updates as you speak into your microphone, or upload an audio file, or pick one of three sample clips (two short speeches and a 60-second TED excerpt), then press Transcribe. The clip stays in place afterwards, so you can transcribe the same audio again with the other model and compare.",
    "Whisper base, 74 million parameters, is the default. It handles 99 languages and returns timestamped segments, shown as minutes and seconds beside each line. It is a 146 MB download on WebGPU and 232 MB on the CPU, where the decoder has to stay at full precision to load at all. Moonshine tiny, 27 million parameters and 86 MB on the CPU, is English only and is the low-latency choice for live captions. Audio is converted to 16 kHz mono before the model sees it.",
    "Live transcription re-reads the most recent 30 seconds of the take every 1.5 seconds. Whisper tends to invent words in silence, often a lone \"you\" in a quiet room, so an optional switch skips those updates when no new speech has arrived and the page counts how many it skipped. The pass when you press Stop always runs.",
    "The page transcribes; it does not say who is speaking. The model runs on your GPU through WebGPU, or on the CPU through WebAssembly, and your voice and your files never leave the device.",
  ],
  aliases: [
    "speech to text",
    "automatic speech recognition",
    "Whisper",
    "voice-to-text",
    "transcribe",
    "live transcription",
    "Moonshine",
  ],
};
