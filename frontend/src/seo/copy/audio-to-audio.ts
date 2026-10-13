import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Remove background noise from speech with DeepFilterNet3, at its native 48 kHz.",
  about: [
    "Speech enhancement, or noise reduction, takes a recording of someone talking over background noise and returns the same speech with the noise suppressed. Here you upload a clip, record six seconds from your microphone or pick a sample, press Enhance, and the page shows the original and the cleaned version as two waveforms, one above the other. Play each in turn and listen to the noise floor, then download the denoised audio as a WAV file. The bundled samples are clean studio speech, so the useful test is a noisy recording of your own.",
    "The model is DeepFilterNet3, 2.3 million parameters and a 9 MB download. It works at 48 kHz, unlike the other pages here that take audio in, which convert it to 16 kHz: resampling down would throw away the high frequencies it is trained to repair. There is no ready-made pipeline for it, so the page runs the ONNX graph through ONNX Runtime Web directly, and the signal processing around the network — the short-time Fourier transform, the ERB filterbank, the deep filtering and the overlap-add — is written for this app and checked against the official libDF implementation.",
    "It runs in a Web Worker on your GPU through WebGPU where available, with a WebAssembly fallback on the CPU. The audio never leaves your machine. It is a denoiser for voice, not a stem splitter: separating vocals from music needs a large model such as Demucs that wants the whole track at once, and that stays on a server.",
  ],
  aliases: [
    "remove background noise",
    "noise reduction",
    "speech enhancement",
    "denoise",
    "DeepFilterNet",
  ],
};
