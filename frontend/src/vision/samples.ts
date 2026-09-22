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

/**
 * Pictures whose subject is **printed text**, for `/image-to-text`.
 *
 * Kept out of `IMAGE_SAMPLES` on purpose: an advertisement is a poor sample for
 * a detector or a depth model, and a sample row is a set of suggestions, not a
 * catalogue. The OCR route appends these to the shared five.
 *
 * The `@slow` spec asserts a substring of what the model reads back off the
 * Coca-Cola advertisement — **change that assertion if this sample changes**,
 * because "some text came back" would pass on a broken processor path that
 * hands the model noise.
 */
export const TEXT_SAMPLES: ImageSample[] = [
  {
    id: "coca-cola",
    label: "Advertisement",
    url: `${BASE}/coca_cola_advertisement.png`,
    hint: "Large printed text on a busy background — the OCR case",
    expect: "the brand name, and most of the surrounding copy",
  },
  {
    id: "book-cover",
    label: "Book cover",
    url: `${BASE}/book-cover.png`,
    hint: "Title, author, publisher — several sizes of type at once",
    expect: "the title, at minimum",
  },
];

/**
 * Photographed **documents**, for `/document-question-answering`.
 *
 * Kept separate from `TEXT_SAMPLES` because the two answer different questions.
 * Those are pictures that happen to contain type — an advertisement, a book
 * cover — and they are there to exercise OCR. These are documents with
 * *structure*: fields, labels and values, which is what a document-QA model is
 * asked about ("what is the invoice number", not "what does this say").
 *
 * `invoice.png` is the Transformers.js docs' own DocVQA example, which makes it
 * the natural fixture for the `@slow` spec: its invoice number is a known answer
 * on a known input, and "some text appeared" would pass without it.
 */
export const DOCUMENT_SAMPLES: ImageSample[] = [
  {
    id: "invoice",
    label: "Invoice",
    url: `${BASE}/invoice.png`,
    hint: "Fields and values — ask for the invoice number, a date, or the total",
    expect: "the exact field you asked for, copied from the page",
  },
  {
    id: "receipt",
    label: "Receipt",
    url: `${BASE}/receipt.png`,
    hint: "Denser and less regular than the invoice — the harder of the two",
    expect: "a line item or a total, when the print is legible enough",
  },
];

/**
 * Pictures whose subject is **a person**, for `/background-removal`.
 *
 * Kept separate from `IMAGE_SAMPLES` for a reason that is a fact about the
 * models rather than about taste: MODNet — the default, and the permissively
 * licensed one — is a *portrait* matting model. Asked to matte a photograph of
 * a car it does not fail loudly; it returns an almost empty matte, and the page
 * shows an empty checkerboard. (The `@slow` spec measured 0.2% coverage on the
 * beetle before these existed, which is how the gap was found.)
 *
 * So the matting route leads with these and offers the general set after. The
 * `@slow` spec asserts a coverage band on the portrait — **change that
 * assertion if this sample changes.**
 */
export const PORTRAIT_SAMPLES: ImageSample[] = [
  {
    id: "portrait",
    label: "Portrait",
    url: `${BASE}/portrait-of-woman.jpg`,
    hint: "One person against a plain background — what portrait matting is built for",
    expect: "a clean cut-out covering roughly the middle third of the frame",
  },
  {
    id: "afro",
    label: "Hair detail",
    url: `${BASE}/woman-with-afro.jpg`,
    hint: "Fine hair against a busy background — where a soft matte earns its keep",
    expect: "strands that fade rather than a jagged silhouette",
  },
];
