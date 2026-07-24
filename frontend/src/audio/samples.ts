// Known-good speech clips for sanity-checking the ASR model from the UI. Each
// has a reference transcript so a working model's output can be eyeballed
// against it — if the model returns garbage on JFK, the model (not your mic) is
// the problem. Hosted on the same Hugging Face origin the ONNX weights come from
// (CORS-enabled), so no binaries live in the repo; they're the canonical
// Transformers.js demo clips.

export interface AudioSample {
  id: string;
  label: string;
  url: string;
  /**
   * Roughly what a healthy ASR model transcribes. Punctuation/casing vary by
   * model, so compare on words, not character-for-character.
   */
  reference: string;
  /** Short hint (length / character of the clip). */
  hint: string;
}

const BASE =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";

export const AUDIO_SAMPLES: AudioSample[] = [
  {
    id: "jfk",
    label: "JFK",
    url: `${BASE}/jfk.wav`,
    reference:
      "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    hint: "~11 s · clean studio speech — the classic Whisper smoke test",
  },
  {
    id: "mlk",
    label: "MLK",
    url: `${BASE}/mlk.wav`,
    reference:
      "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
    hint: "short · clear speech",
  },
  {
    id: "ted",
    label: "TED (60 s)",
    url: `${BASE}/ted_60.wav`,
    reference: "",
    hint: "60 s · long-form, exercises chunking + timestamps · ~11 MB download",
  },
];
