import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Turn text into speech entirely in your browser — the model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker.",
  about: [
    "Text to speech, also called speech synthesis or TTS, is a way to read text aloud: you type a sentence and a neural voice model generates the audio for it. On this page you write or paste the text, choose a voice where the model has more than one, press Speak and hear the result straight away. The clip can be replayed or saved as a WAV file, with its length and sample rate shown beside it.",
    "Three models are offered. Kokoro 82M is the default, an 82-million-parameter model built for the browser, with six named voices in US and UK English. MMS English is a small end-to-end VITS model of 36 million parameters, and SpeechT5 is Microsoft's 144-million-parameter model, which takes a fixed speaker embedding instead of a voice list. The picker quotes each download before anything is fetched, and nothing loads until you press the load button; after that the weights are cached by the browser.",
    "Synthesis runs in a Web Worker, on WebGPU when your browser offers it and on the CPU through WebAssembly when it does not. The text you type is never sent to a server, so this works as a private, free text to speech tool for anything you would rather not paste into an online service.",
    "What it does not do: there is no voice cloning, and the voices on offer are English. Larger generative audio models such as Bark are too heavy for a browser tab, and music generation was removed because its only browser checkpoint, MusicGen, is a 599 MB download at its smallest.",
  ],
  aliases: [
    "text to speech",
    "speech synthesis",
    "TTS",
    "read text aloud",
    "Kokoro",
    "private, free text to speech",
  ],
};
