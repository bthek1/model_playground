import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Identify the sounds in an audio clip, or score it against labels you write yourself, in your browser.",
  about: [
    "Audio classification, also called sound classification, listens to a clip and names what is in it: speech, music, a dog, an engine. On this page you upload an audio file, record five seconds from your microphone or choose a sample, then press Classify to get a ranked list of labels with a percentage for each. The clip is converted to 16 kHz mono and held, so you can classify it again with a different model without loading it twice.",
    "Three models are on offer. AST, the Audio Spectrogram Transformer fine-tuned on AudioSet, has 87 million parameters and 527 fixed sound-event tags covering music, speech, animals and machines. The wav2vec2 keyword spotting model, 95 million parameters, recognises short spoken commands such as yes, no, up and down. CLAP, 153 million parameters, does zero-shot audio classification: instead of a fixed label set it scores the clip against text prompts you type, one per line, such as \"a dog barking\" or \"rain falling\". Editing the prompts runs nothing; the next press of Classify re-scores the same clip.",
    "The model runs in a Web Worker on your GPU through WebGPU, or on the CPU through WebAssembly, and the picker quotes the download before anything is fetched. Your recording is never uploaded.",
    "Each run gives one ranking for the whole clip, not a timeline of when each sound occurs. CLAP's percentages compare your prompts with one another, so they show which description fits best rather than whether any of them is right.",
  ],
  aliases: [
    "audio classification",
    "sound classification",
    "AudioSet",
    "zero-shot audio classification",
    "keyword spotting",
    "CLAP",
    "identify the sounds",
  ],
};
