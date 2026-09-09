// Known-good images for exercising a vision model from the UI, so every route
// works before the user has picked a file or granted a camera permission. The
// vision counterpart of `audio/samples.ts`.
//
// Hosted on the same Hugging Face origin the ONNX weights come from (CORS
// enabled), so no binaries live in the repo — these are the canonical
// Transformers.js demo images. Each URL was checked for a 200 before it was
// added; `expect` says roughly what a healthy model should say about it, which
// is what makes a sample a smoke test rather than decoration.

export interface ImageSample {
  id: string;
  label: string;
  url: string;
  /** Short description of the picture, shown in the picker. */
  hint: string;
  /** Roughly what a working model should report. Compare loosely. */
  expect: string;
}

const BASE =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";

export const IMAGE_SAMPLES: ImageSample[] = [
  {
    id: "cats",
    label: "Cats",
    url: `${BASE}/cats.jpg`,
    hint: "Two cats on a sofa — the Transformers.js house image",
    expect: "a cat breed (tabby, Egyptian cat…) near the top",
  },
  {
    id: "tiger",
    label: "Tiger",
    url: `${BASE}/tiger.jpg`,
    hint: "One animal, filling the frame — the easy case",
    expect: "tiger, with a confident margin over everything else",
  },
  {
    id: "football",
    label: "Football match",
    url: `${BASE}/football-match.jpg`,
    hint: "Several people, a ball, a crowd — the cluttered case",
    expect: "a scene or sport label, and a much flatter score distribution",
  },
  {
    id: "city",
    label: "City street",
    url: `${BASE}/city-streets.jpg`,
    hint: "Cars, people, buildings — the detection/segmentation workhorse",
    expect: "a street-scene label; no single object owns the picture",
  },
  {
    id: "beetle",
    label: "Beetle (car)",
    url: `${BASE}/beetle.png`,
    hint: "One object against a plain background",
    expect: "a car label — convertible, beach wagon, or similar",
  },
];

export const DEFAULT_IMAGE_SAMPLE = IMAGE_SAMPLES[0].id;
