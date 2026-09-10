// The bundled gallery for /image-features, so the page is useful before the
// user has uploaded anything.
//
// Twelve pictures rather than five, and chosen as **clusters** rather than for
// variety alone: three animals, three city scenes, two portraits, two
// landscapes, two objects. A similarity search over a set with no near
// neighbours in it has nothing to show — every result is "the least unrelated
// thing", and the page teaches nothing. The clusters are also what makes the
// `@slow` assertion possible: query with an animal and the top neighbour must be
// an animal, not a street.
//
// Same origin and same CORS story as `samples.ts` — the canonical Transformers.js
// demo images, checked for a 200 before being added, so no binaries live in the
// repo. `group` is shown in the UI *and* is what the E2E spec asserts on; it is
// the ground truth this page is scored against.

export interface GalleryImage {
  id: string;
  label: string;
  url: string;
  /** The cluster this picture belongs to. Ground truth for the neighbours. */
  group: "animal" | "city" | "person" | "landscape" | "object";
}

const BASE =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";

export const GALLERY_IMAGES: GalleryImage[] = [
  { id: "corgi", label: "Corgi", url: `${BASE}/corgi.jpg`, group: "animal" },
  { id: "cats", label: "Cats", url: `${BASE}/cats.jpg`, group: "animal" },
  {
    id: "butterfly",
    label: "Butterfly",
    url: `${BASE}/butterfly.jpg`,
    group: "animal",
  },
  {
    id: "city-streets",
    label: "City street",
    url: `${BASE}/city-streets.jpg`,
    group: "city",
  },
  {
    id: "new-york",
    label: "New York",
    url: `${BASE}/new-york.jpg`,
    group: "city",
  },
  { id: "airport", label: "Airport", url: `${BASE}/airport.jpg`, group: "city" },
  {
    id: "portrait",
    label: "Portrait",
    url: `${BASE}/portrait-of-woman.jpg`,
    group: "person",
  },
  {
    id: "afro",
    label: "Portrait, outdoors",
    url: `${BASE}/woman-with-afro.jpg`,
    group: "person",
  },
  {
    id: "moraine",
    label: "Mountain lake",
    url: `${BASE}/moraine-lake.png`,
    group: "landscape",
  },
  {
    id: "savanna",
    label: "Savanna",
    url: `${BASE}/savanna.jpg`,
    group: "landscape",
  },
  { id: "beetle", label: "Beetle", url: `${BASE}/beetle.png`, group: "object" },
  { id: "house", label: "House", url: `${BASE}/house.jpg`, group: "object" },
];

/**
 * Roughly what the gallery costs to embed, for the copy above the button.
 *
 * Twelve forward passes, one at a time — the page never runs two, because two
 * overlapping calls into one ONNX session is not a guarantee worth relying on.
 */
export const GALLERY_SIZE = GALLERY_IMAGES.length;
