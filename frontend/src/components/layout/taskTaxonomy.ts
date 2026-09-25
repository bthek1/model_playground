import type { LucideIcon } from "lucide-react";
import {
  AudioLines,
  Eye,
  Gamepad2,
  Layers,
  Shapes,
  Sigma,
  Table,
  Type,
} from "lucide-react";

/**
 * The app's navigation taxonomy — a two-level tree of task categories and the
 * tasks within them, modelled on the Hugging Face pipeline taxonomy.
 *
 * The sidebar is data-driven: to add a task, add an entry here. A task links to
 * a real route when one exists (`to`), otherwise it falls back to the generic
 * `/tasks/$slug` landing page (see `routes/tasks.$slug.tsx`).
 */

export interface TaskItem {
  label: string;
  /** Kebab-case identifier derived from `label`; used in `/tasks/$slug`. */
  slug: string;
  /** Route this task links to. Real route if implemented, else `/tasks/$slug`. */
  to: string;
}

export interface TaskCategory {
  label: string;
  icon: LucideIcon;
  tasks: TaskItem[];
}

function slugify(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Tasks that map to a real, implemented route. Everything else routes to the
 * generic `/tasks/$slug` placeholder.
 */
const REAL_ROUTES: Record<string, string> = {
  "gpu-playground": "/playground",
  "linear-model-training": "/training",
  "tensor-arithmetic": "/tensor",
  "automatic-speech-recognition": "/asr",
  "audio-classification": "/audio-classification",
  "text-to-speech": "/text-to-speech",
  "audio-to-audio": "/audio-to-audio",
  "voice-activity-detection": "/vad",
  "image-classification": "/image-classification",
  "depth-estimation": "/depth",
  "object-detection": "/object-detection",
  "image-segmentation": "/segmentation",
  "zero-shot-image-classification": "/zero-shot-image-classification",
  "zero-shot-object-detection": "/zero-shot-object-detection",
  "image-feature-extraction": "/image-features",
  "mask-generation": "/mask-generation",
  "keypoint-detection": "/pose",
  "video-classification": "/video-classification",
  "background-removal": "/background-removal",
  // The route covers the **super-resolution subset** of image-to-image only.
  // Editing / img2img is diffusion and stays on a server (vision.md §3.12), and
  // the page says so in its own header rather than letting the slug over-promise.
  "image-to-image": "/super-resolution",
  // Likewise the **depth-to-point-cloud subset** only: full reconstruction is
  // SD-derived and stays on a server. The route adds no new model — it is
  // /depth's checkpoint plus an unprojection.
  "image-to-3d": "/image-to-3d",
  // The Tabular category, and the only pages in the app that download nothing
  // at all: there is no checkpoint to fetch because the model is *fitted in the
  // tab* on the user's own CSV. That inverts the usual privacy sentence — every
  // other route says "the weights come to you", these say "your file never
  // leaves the device" — and it is the stronger claim, because tabular data is
  // the kind people actually mind about.
  "tabular-classification": "/tabular-classification",
  // The only route on the raw-WebGPU path outside Theory: no checkpoint exists,
  // so the network is written as WGSL and trained in the tab. Covers node
  // classification and the oversmoothing demonstration; link prediction and
  // graph classification are still open on the roadmap.
  "graph-machine-learning": "/graph",
  // The other half of the Graph ML roadmap's shipped work: the same Cora and
  // the same kernels, trained on a graph with 15% of its citations removed and
  // asked to score the pairs it never saw. Separate from /graph because it is a
  // different question — a different split, loss, metric and picture — and one
  // page per question is worth more than the reuse.
  "link-prediction": "/link-prediction",
  // The last of the Graph ML roadmap's four sections, and the only page in the
  // repo that downloads a *dataset* rather than a checkpoint: 1113 protein
  // graphs from the Hub, one label each.
  "graph-classification": "/graph-classification",
  // The Multimodal category's first route, and the app's first vision-language
  // model. Unlike the two subset routes above, this one covers its slug whole:
  // an image and a question in, prose out. Visual Question Answering (§3.2) will
  // share this engine behind its own route rather than folding into this one —
  // a user looking for VQA does not think to click "image-text-to-text".
  "image-text-to-text": "/image-text-to-text",
  // The second Multimodal route, and the cheapest in the repo: it downloads
  // nothing new and adds no engine. VQA in a browser *is* prompting a general
  // VLM — neither classical VQA model (ViLT, BLIP-VQA) has an ONNX export — so
  // this is the route above with the question shaped differently. Separate
  // because a user looking for VQA does not click "Image Text to Text", and the
  // Hub has both tags.
  "visual-question-answering": "/visual-question-answering",
  // The third, and the one that needed a model rather than only a prompt:
  // SmolVLM2's video-instruct tune, 189 MB, on the same engine. A
  // video-language model in a tab is a frame sampler plus an image model, which
  // the page states next to its result rather than implying a temporal
  // understanding it does not have.
  "video-text-to-text": "/video-text-to-text",
  // The Natural Language Processing category's first route, and the smallest
  // useful page in the app: a string in, a score list out, no decode step at
  // all. It establishes `src/text/` — the generic text worker, engine and
  // catalogue the rest of the category rides on.
  "text-classification": "/text-classification",
  // §3.2, and the page where "it runs in your browser" stops being a
  // performance claim: redacting a document you are not allowed to upload is a
  // real reason to want the model on this side of the wire. It also builds
  // `SpanOverlay`, which /question-answering and /fill-mask then reuse.
  "token-classification": "/token-classification",
  // §3.3, and the category's first route that does **not** ride the generic
  // text worker. The `question-answering` pipeline returns `{ answer, score }`
  // and throws away the token indices it chose, so it cannot say *where* in the
  // passage the answer is — which is this page's whole output. `text/qa/` drives
  // the tokenizer and model directly and recovers the character range.
  "question-answering": "/question-answering",
  // §3.4, and the first page in the app whose *label set* is the user's rather
  // than the checkpoint's. Its cost model is the thing it has to say out loud:
  // an NLI model runs once per label, so the page derives the pass count from
  // the label list as it is edited.
  "zero-shot-classification": "/zero-shot-classification",
  // §3.9, and the only page in the category whose models are *base* encoders:
  // masked language modelling is the objective they were pretrained on, so the
  // head is the real one. Four entries across three tokenizer families, so the
  // mask-token hazard is one click away rather than theoretical — the token is
  // inserted by a button, rewritten on a model change, and reconciled against
  // the loaded tokenizer in the engine.
  "fill-mask": "/fill-mask",
  // §3.7, shipped as a pair because the second page is the first with a cosine
  // on the end: one engine, one catalogue, one hook, two taxonomy rows. They
  // stay two routes for the same reason /visual-question-answering is not
  // folded into /image-text-to-text — two Hub tags and two questions.
  //
  // `feature-extraction` maps to `/text-features` rather than the slug's own
  // name, mirroring `/image-features`.
  // §3.5, and the category's first seq2seq page. One language pair at a time,
  // because a Marian checkpoint *is* its direction — so the direction control
  // is a model selector in SELECT, and switching it is another ~200 MB
  // download rather than a toggle.
  "translation": "/translation",
  // §3.6, and the page that exists because of one measurement: DistilBART opens
  // a q8 session on WebGPU (283.9 MB) where every other configuration is over
  // the size bar, and its q8 WASM session does not open at all. T5-small is the
  // floor that keeps a CPU path. The lead-3 baseline ships as the output's empty
  // state — it needs no model, and beating it is harder than it sounds.
  "summarization": "/summarization",
  // §3.8, the category's only streaming page — and the one that takes this row
  // back from `/playground`. `/playground` is a WebGPU demo surface, not a task
  // page, and it should not claim a task row; it stays reachable on its own
  // terms. The page is the **decoding strategies made interactive** rather than
  // a chatbot: greedy against sampling on the same prompt, with the repetition
  // loop visible. It is also the payoff for #30 putting `partial` in the shared
  // `ModelResponse` envelope rather than in a private VLM protocol.
  "text-generation": "/text-generation",
  "feature-extraction": "/text-features",
  "sentence-similarity": "/sentence-similarity",
  // §3.10, and the most complete page the category has: **all four retrieval
  // stages run client-side** over a corpus the user pastes in — BM25, dense
  // embeddings, hybrid RRF, and a cross-encoder rerank. Two of the four need no
  // model at all. It is also the second route to hold two models live at once,
  // after /pose, and for the same reason: neither half is useful alone.
  "text-ranking": "/text-ranking",
  // Three slugs deliberately have no route and fall through to the placeholder,
  // for one reason each — every checkpoint the task has is too heavy to offer:
  //
  //   text-to-audio                 MusicGen-small, the only browser music
  //                                 model, is 599 MB on WASM / 1127 MB on
  //                                 WebGPU — past the ~1 GB ceiling.
  //   image-to-text                 Florence-2 (544 MB) and vit-gpt2 (482 MB)
  //                                 are the whole catalogue; neither is a
  //                                 download to put behind a sidebar click.
  //   document-question-answering   Donut is 411 MB on WebGPU and 597 MB on
  //                                 WASM (its decoder cannot be quantized —
  //                                 see `model/backend.ts`), and the pipeline
  //                                 hardcodes Donut's prompt, so there is no
  //                                 lighter second entry to fall back to.
  //
  // All three are written up in their category roadmaps with the measured
  // numbers, which `adding-a-task-page.md` §0 counts as finished work.
};

function task(label: string): TaskItem {
  const slug = slugify(label);
  return { label, slug, to: REAL_ROUTES[slug] ?? `/tasks/${slug}` };
}

function tasks(labels: string[]): TaskItem[] {
  return labels.map(task);
}

export const taskCategories: TaskCategory[] = [
  {
    label: "Audio",
    icon: AudioLines,
    tasks: tasks([
      "Text to Speech",
      "Text to Audio",
      "Automatic Speech Recognition",
      "Audio to Audio",
      "Audio Classification",
      "Voice Activity Detection",
    ]),
  },
  {
    label: "Computer Vision",
    icon: Eye,
    tasks: tasks([
      "Depth Estimation",
      "Image Classification",
      "Object Detection",
      "Image Segmentation",
      "Text to Image",
      "Image to Text",
      "Image to Image",
      "Image to Video",
      "Unconditional Image Generation",
      "Video Classification",
      "Text to Video",
      "Zero Shot Image Classification",
      "Mask Generation",
      "Zero Shot Object Detection",
      "Text to 3D",
      "Image to 3D",
      "Image Feature Extraction",
      "Keypoint Detection",
      "Video to Video",
      // **A deliberate departure from the Hub's task list.** Every other row in
      // this category mirrors a Hugging Face pipeline tag; the Hub has no
      // `background-removal` task — Transformers.js added the pipeline itself,
      // as a subclass of image segmentation. It is listed anyway because the
      // page stands alone as useful and an unlisted route is one nobody finds.
      // See #24 for the call.
      "Background Removal",
    ]),
  },
  {
    label: "Multimodal",
    icon: Layers,
    tasks: tasks([
      "Audio Text to Text",
      "Image Text to Text",
      "Image Text to Image",
      "Image Text to Video",
      "Visual Question Answering",
      "Document Question Answering",
      "Video Text to Text",
      "Visual Document Retrieval",
      "Any to Any",
    ]),
  },
  {
    label: "Natural Language Processing",
    icon: Type,
    tasks: tasks([
      "Text Classification",
      "Token Classification",
      "Table Question Answering",
      "Question Answering",
      "Zero Shot Classification",
      "Translation",
      "Summarization",
      "Feature Extraction",
      "Text Generation",
      "Fill Mask",
      "Sentence Similarity",
      "Text Ranking",
    ]),
  },
  {
    label: "Other",
    icon: Shapes,
    // "Link Prediction" is a row the Hub does not have — its taxonomy stops at
    // graph-ml — added for the same reason Computer Vision has a
    // Background Removal row: a shipped route that answers its own question
    // deserves its own entry rather than being buried inside a neighbour's.
    tasks: tasks([
      "Graph Machine Learning",
      "Link Prediction",
      "Graph Classification",
    ]),
  },
  {
    label: "Reinforcement Learning",
    icon: Gamepad2,
    tasks: tasks(["Reinforcement Learning", "Robotics"]),
  },
  {
    label: "Tabular",
    icon: Table,
    tasks: tasks([
      "Tabular Classification",
      "Tabular Regression",
      "Time Series Forecasting",
    ]),
  },
  {
    label: "Theory",
    icon: Sigma,
    tasks: tasks([
      "Linear Model Training",
      "Discrete Maths",
      "Tensor Arithmetic",
      // Taking §3.8's task row away from `/playground` would have left it
      // reachable only by typing the URL, which is not what "stays reachable on
      // its own terms" means. Theory is this repo's own non-Hub category and is
      // where the hand-written WGSL surfaces already live, so the demo gets a
      // row of its own here instead of borrowing an NLP task's.
      "GPU Playground",
    ]),
  },
];

/** Flat lookup of every task by slug — used by the `/tasks/$slug` route. */
export const tasksBySlug: Record<string, TaskItem> = Object.fromEntries(
  taskCategories.flatMap((c) => c.tasks.map((t) => [t.slug, t])),
);

/** The category label that owns the given pathname, if any. */
export function categoryForPath(pathname: string): string | undefined {
  for (const category of taskCategories) {
    if (category.tasks.some((t) => t.to === pathname)) return category.label;
  }
  return undefined;
}
