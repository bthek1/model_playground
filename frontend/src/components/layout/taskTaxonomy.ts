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
 * The sidebar is data-driven: to add a task, add an entry here **and** its
 * route in `REAL_ROUTES`. The sidebar lists only tasks that have a page — a
 * Hub task with no route is not listed at all (#59), and `task()` throws on a
 * label with no route so an unbuilt row cannot creep back in. Why each missing
 * Hub task has no page is recorded in its category roadmap under `docs/roadmaps/`.
 */

export interface TaskItem {
  label: string;
  /** Kebab-case identifier derived from `label`; the key into `REAL_ROUTES`. */
  slug: string;
  /** The implemented route this task links to. */
  to: string;
  /**
   * One sentence, 70–160 characters: what the page does and what runs it.
   * It is the page's meta description and its line in the home page's task
   * index (#64, #65), so write it for someone choosing a page, not for us.
   */
  description: string;
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

/** Every sidebar task's implemented route, keyed by slug. */
const REAL_ROUTES: Record<string, string> = {
  "gpu-playground": "/playground",
  "linear-model-training": "/training",
  "tensor-arithmetic": "/tensor",
  // Theory's last row, and one topic rather than a syllabus: walks in a graph,
  // and why a k-layer GNN reads exactly the nodes within k hops. No checkpoint
  // and no dataset — the same shape as /tensor.
  "discrete-maths": "/discrete-maths",
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
  "tabular-regression": "/tabular-regression",
  // The one route in the app with no model, no download and no worker: the
  // baselines are the page, and the two foundation forecasters worth wanting
  // publish no ONNX weights at all.
  "time-series-forecasting": "/time-series-forecasting",
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
  // The Reinforcement Learning category, and the only pages whose subject is a
  // loop the user watches. `/rl` trains in the tab on the CPU — Phase 0 of #51
  // measured that the GPU round trip costs more than an RL step's arithmetic —
  // so it has no checkpoint, no download and no LOAD band.
  "reinforcement-learning": "/rl",
  // The perception half of robot learning: an open-vocabulary detector and a
  // depth model the app already ships, composed as one pair in two workers.
  // The control half does not port (no simulator, no hardware), and the page
  // says so beside its result.
  "robotics": "/robotics",
};

function task([label, description]: TaskEntry): TaskItem {
  const slug = slugify(label);
  const to = REAL_ROUTES[slug];
  // A row without a page is a promise the sidebar cannot keep: it used to fall
  // through to a "/tasks/$slug — on the roadmap" placeholder, for tasks the
  // roadmaps had measured and ruled out. Fail at import rather than list one.
  if (!to) throw new Error(`Task "${label}" has no route in REAL_ROUTES`);
  return { label, slug, to, description };
}

/** A sidebar row as written below: its label and its one-line description. */
type TaskEntry = readonly [label: string, description: string];

function tasks(entries: TaskEntry[]): TaskItem[] {
  return entries.map(task);
}

export const taskCategories: TaskCategory[] = [
  {
    label: "Audio",
    icon: AudioLines,
    tasks: tasks([
      [

        "Text to Speech",

        "Turn text into speech in your browser with Kokoro, MMS and SpeechT5 — synthesised on your own GPU or CPU, with nothing sent to a server.",

      ],
      [
        "Automatic Speech Recognition",
        "Transcribe speech to text in your browser with Whisper and Moonshine — live from the microphone or from a file, privately, on your own device.",
      ],
      [
        "Audio to Audio",
        "Remove background noise from speech in your browser with DeepFilterNet3 — full-band 48 kHz enhancement that never uploads your recording.",
      ],
      [
        "Audio Classification",
        "Tag sounds in your browser with AST, wav2vec2 and CLAP — fixed labels or your own text prompts, scored on your device with nothing uploaded.",
      ],
      [
        "Voice Activity Detection",
        "Find where speech starts and stops in a recording with Silero VAD, running in your browser on the CPU at around 100x real time.",
      ],]),
  },
  {
    label: "Computer Vision",
    icon: Eye,
    tasks: tasks([
      [

        "Depth Estimation",

        "Estimate depth from a single photo in your browser with Depth Anything — a relative depth map computed on your own GPU, never uploaded.",

      ],
      [
        "Image Classification",
        "Classify photos in your browser with ViT, ResNet and MobileNet — from a file or your webcam, with ranked labels computed on your device.",
      ],
      [
        "Object Detection",
        "Detect and box objects in photos or a live webcam feed in your browser with D-FINE, RT-DETR, YOLOS and DETR running on your own GPU.",
      ],
      [
        "Image Segmentation",
        "Segment images into labelled regions in your browser with SegFormer and DETR panoptic — masks computed on your device, with an overlay.",
      ],
      [
        "Image to Image",
        "Upscale images with Swin2SR super-resolution in your browser, tiled so large pictures fit, and compared against a bicubic baseline.",
      ],
      [
        "Video Classification",
        "Classify video clips in your browser by sampling frames through an image model — a frame-level baseline running on your own device.",
      ],
      [
        "Zero Shot Image Classification",
        "Classify images against labels you type with CLIP and SigLIP in your browser — no training, and the label embeddings are cached.",
      ],
      [
        "Mask Generation",
        "Click on an image to segment any object with Segment Anything (SAM) in your browser — encode once, then decode a mask per click.",
      ],
      [
        "Zero Shot Object Detection",
        "Detect objects by name with OWL-ViT and OWLv2 in your browser — type what to look for and get boxes without training a detector.",
      ],
      [
        "Image to 3D",
        "Turn a single photo into an interactive 3D point cloud in your browser — Depth Anything depth, unprojected and rendered with WebGPU.",
      ],
      [
        "Image Feature Extraction",
        "Extract image embeddings with CLIP, DINOv2 and DINOv3 in your browser and compare pictures by cosine similarity, on your own device.",
      ],
      [
        "Keypoint Detection",
        "Estimate human pose in your browser — detect each person, then place body keypoints and a skeleton, from a photo or your webcam.",
      ],
      // **A deliberate departure from the Hub's task list.** Every other row in
      // this category mirrors a Hugging Face pipeline tag; the Hub has no
      // `background-removal` task — Transformers.js added the pipeline itself,
      // as a subclass of image segmentation. It is listed anyway because the
      // page stands alone as useful and an unlisted route is one nobody finds.
      // See #24 for the call.
      [
        "Background Removal",
        "Remove the background from a photo in your browser with MODNet or RMBG — a soft alpha matte and a transparent PNG, never uploaded.",
      ],]),
  },
  {
    label: "Multimodal",
    icon: Layers,
    tasks: tasks([
      [

        "Image Text to Text",

        "Ask questions about an image with SmolVLM, a vision-language model running in your browser on WebGPU, with the answer streamed as it is written.",

      ],
      [
        "Visual Question Answering",
        "Visual question answering in your browser — ask a short question about a picture and get a one-word answer from a small VLM on WebGPU.",
      ],
      [
        "Video Text to Text",
        "Ask a vision-language model about a video in your browser — SmolVLM2 reads frames sampled from the clip on WebGPU and answers in text.",
      ],]),
  },
  {
    label: "Natural Language Processing",
    icon: Type,
    tasks: tasks([
      [

        "Text Classification",

        "Classify text in your browser — sentiment and financial-tone models side by side, with the full score list, running on your own device.",

      ],
      [
        "Token Classification",
        "Find names, places and organisations with named-entity recognition in your browser, and redact them without uploading the document.",
      ],
      [
        "Question Answering",
        "Extractive question answering in your browser — paste a passage, ask a question, and see the exact answer span highlighted in the text.",
      ],
      [
        "Zero Shot Classification",
        "Classify text against labels you type with an NLI model in your browser — no training, an editable hypothesis template, one pass per label.",
      ],
      [
        "Translation",
        "Translate text with Marian models in your browser — one language pair per model, private machine translation on your own device.",
      ],
      [
        "Summarization",
        "Summarize articles in your browser with T5 and DistilBART, measured against a lead-3 baseline, with a sentence-level faithfulness check.",
      ],
      [
        "Feature Extraction",
        "Turn sentences into embeddings in your browser with MiniLM, BGE and Nomic models — inspect the vectors and try Matryoshka truncation.",
      ],
      [
        "Text Generation",
        "Generate text with small language models such as SmolLM2 and GPT-2 in your browser — compare greedy decoding with sampling, streamed live.",
      ],
      [
        "Fill Mask",
        "Predict a masked word with BERT, RoBERTa and ModernBERT in your browser — and probe what each model's training corpus taught it.",
      ],
      [
        "Sentence Similarity",
        "Compare sentences by meaning in your browser — embedding models score paraphrases against unrelated text with cosine similarity.",
      ],
      [
        "Text Ranking",
        "Rank documents for a query in your browser with BM25, dense embeddings, hybrid fusion and a cross-encoder reranker, side by side.",
      ],]),
  },
  {
    label: "Other",
    icon: Shapes,
    // "Link Prediction" is a row the Hub does not have — its taxonomy stops at
    // graph-ml — added for the same reason Computer Vision has a
    // Background Removal row: a shipped route that answers its own question
    // deserves its own entry rather than being buried inside a neighbour's.
    tasks: tasks([
      [

        "Graph Machine Learning",

        "Train a graph neural network (GCN, GraphSAGE, GIN, GAT) on the Cora citation graph in your browser with WebGPU, and watch oversmoothing.",

      ],
      [
        "Link Prediction",
        "Predict missing citations in the Cora graph with a GNN trained in your browser on WebGPU, scored by AUC on held-out edges.",
      ],
      [
        "Graph Classification",
        "Classify protein graphs from the PROTEINS dataset with a graph neural network trained in your browser, beside a majority baseline.",
      ],]),
  },
  {
    label: "Reinforcement Learning",
    icon: Gamepad2,
    tasks: tasks([
      ["Reinforcement Learning", "Watch Q-learning, REINFORCE and Actor-Critic learn FrozenLake and CartPole in your browser, checked against exact value iteration."],
      ["Robotics", "Robot perception in your browser — find objects by name with OWLv2 and order them by depth — plus a behaviour-cloning experiment."],
    ]),
  },
  {
    label: "Tabular",
    icon: Table,
    tasks: tasks([
      [

        "Tabular Classification",

        "Fit classifiers to your own CSV in your browser — logistic regression, random forest, gradient boosting and an MLP, with nothing uploaded.",

      ],
      [
        "Tabular Regression",
        "Fit regression models to your own CSV in your browser — ridge, trees, boosting and quantile bands, with residual plots and nothing uploaded.",
      ],
      [
        "Time Series Forecasting",
        "Forecast a time series in your browser with naive, seasonal and drift baselines, scored by a rolling-origin backtest and MASE.",
      ],]),
  },
  {
    label: "Theory",
    icon: Sigma,
    tasks: tasks([
      [

        "Linear Model Training",

        "Train a softmax classifier on MNIST in your browser with hand-written WebGPU kernels, and watch its weights turn into digit templates.",

      ],
      [
        "Discrete Maths",
        "Step a breadth-first search and count walks with powers of the adjacency matrix — why a k-layer graph network reads only k hops.",
      ],
      [
        "Tensor Arithmetic",
        "Run matrix and tensor operations on your GPU with raw WebGPU compute shaders in a Web Worker — edit the operands and compute.",
      ],
      // Taking §3.8's task row away from `/playground` would have left it
      // reachable only by typing the URL, which is not what "stays reachable on
      // its own terms" means. Theory is this repo's own non-Hub category and is
      // where the hand-written WGSL surfaces already live, so the demo gets a
      // row of its own here instead of borrowing an NLP task's.
      [
        "GPU Playground",
        "Pick a model and run it on custom WebGPU kernels in your browser — the raw-WGSL inference workspace behind the task pages.",
      ],]),
  },
];

/** The category label that owns the given pathname, if any. */
export function categoryForPath(pathname: string): string | undefined {
  for (const category of taskCategories) {
    if (category.tasks.some((t) => t.to === pathname)) return category.label;
  }
  return undefined;
}
