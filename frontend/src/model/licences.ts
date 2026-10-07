// Every model licence, keyed by the Hub repo the browser downloads (#62).
//
// A licence belongs to a **repo**, not to a catalogue entry: the pose and
// ranking entries are pairs, the grounding entry is assembled from two other
// pages' entries, and a model offered on two pages (CLIP, OWLv2, Depth Anything)
// would otherwise state its licence twice and drift. So the catalogues stay as
// they are, `entryRepos()` says which repos an entry fetches, and this table is
// the one place a licence is written down. `licences.test.ts` fails, naming the
// repo, if any catalogue downloads something this table does not cover.
//
// ---
//
// **A mirror's licence is its upstream's.** Most `Xenova/*` and `onnx-community/*`
// repos are ONNX conversions whose cards declare no licence at all; converting
// weights does not change their terms, so `from` names the repo whose card
// does. `hub` is what that card's `license:` field says, verbatim, and
// `just fe-e2e-models` reads it back off the Hub and fails on a mismatch — the
// same discipline as the mask-token and pooling checks: catalogue data checked
// against the source, not trusted.
//
// Where the card is silent but the authors' own code repository licenses the
// released weights (CLIP, SAM 2, ViTPose, FinBERT), `hub` is `null`, `url`
// points at that repository's LICENSE, and `via` says so in a sentence. Where
// nobody states anything (the two `typeform` NLI heads) the terms are
// `unstated` and the picker says that rather than guessing.
//
// Audited 2026-10-07 against the Hub API and the upstream repositories. Nothing
// here is legal advice; it is what each publisher says about its own model.

import type { CatalogueEntry } from "./catalogue";

/**
 * What a user may do with a model's outputs, in the four cases this app has.
 *
 * - `permissive` — Apache-2.0, MIT, BSD, AFL, CC BY: use freely, with
 *   attribution (which `/licences` provides).
 * - `non-commercial` — research or personal use only.
 * - `custom` — the publisher's own licence, with conditions worth reading.
 * - `unstated` — no licence anywhere; by default that is all rights reserved.
 */
export type LicenceTerms = "permissive" | "non-commercial" | "custom" | "unstated";

export interface ModelLicence {
  /** The licence's name, as its publisher states it (SPDX id where there is one). */
  name: string;
  terms: LicenceTerms;
  /** The licence text, or the page that states it. */
  url: string;
  /** The repo whose card states the licence, when it is not the one downloaded. */
  from?: string;
  /**
   * The `license:` field on the card of `from ?? repo`, verbatim (`null` when
   * the card has none). `just fe-e2e-models` checks it against the Hub.
   */
  hub: string | null;
  /** One sentence, shown wherever the terms are not `permissive`. */
  note?: string;
  /** Where the licence comes from, when it is not the model card. */
  via?: string;
}

const hf = (repo: string) => `https://huggingface.co/${repo}`;

/** A permissive licence declared on the upstream `from`'s card. */
function open(name: string, hub: string, from: string): ModelLicence {
  return { name, terms: "permissive", url: hf(from), hub, from };
}

/** A permissive licence declared on the downloaded repo's own card. */
function own(repo: string, name: string, hub: string): ModelLicence {
  return { name, terms: "permissive", url: hf(repo), hub };
}

const SEGFORMER_LICENCE = "https://github.com/NVlabs/SegFormer/blob/master/LICENSE";

export const MODEL_LICENCES: Readonly<Record<string, ModelLicence>> = {
  // ── Audio ────────────────────────────────────────────────────────────────
  "onnx-community/whisper-base": open("Apache-2.0", "apache-2.0", "openai/whisper-base"),
  "onnx-community/moonshine-tiny-ONNX": own("onnx-community/moonshine-tiny-ONNX", "MIT", "mit"),
  "Xenova/ast-finetuned-audioset-10-10-0.4593": open(
    "BSD-3-Clause",
    "bsd-3-clause",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
  ),
  "Xenova/wav2vec2-base-superb-ks": open("Apache-2.0", "apache-2.0", "superb/wav2vec2-base-superb-ks"),
  "Xenova/clap-htsat-unfused": open("Apache-2.0", "apache-2.0", "laion/clap-htsat-unfused"),
  "onnx-community/Kokoro-82M-v1.0-ONNX": own("onnx-community/Kokoro-82M-v1.0-ONNX", "Apache-2.0", "apache-2.0"),
  "Xenova/mms-tts-eng": {
    name: "CC-BY-NC-4.0",
    terms: "non-commercial",
    url: hf("facebook/mms-tts-eng"),
    from: "facebook/mms-tts-eng",
    hub: "cc-by-nc-4.0",
    note: "Meta's MMS voices are Creative Commons non-commercial: fine for trying here, not for a product.",
  },
  "Xenova/speecht5_tts": open("MIT", "mit", "microsoft/speecht5_tts"),
  "soniqo/DeepFilterNet3-ONNX": own("soniqo/DeepFilterNet3-ONNX", "MIT", "mit"),
  "onnx-community/silero-vad": own("onnx-community/silero-vad", "MIT", "mit"),

  // ── Vision ───────────────────────────────────────────────────────────────
  "Xenova/vit-base-patch16-224": open("Apache-2.0", "apache-2.0", "google/vit-base-patch16-224"),
  "Xenova/resnet-50": open("Apache-2.0", "apache-2.0", "microsoft/resnet-50"),
  "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k": open(
    "Apache-2.0",
    "apache-2.0",
    "timm/mobilenetv4_conv_small.e2400_r224_in1k",
  ),
  "onnx-community/depth-anything-v2-small": own(
    "onnx-community/depth-anything-v2-small",
    "Apache-2.0",
    "apache-2.0",
  ),
  "Xenova/depth-anything-small-hf": open("Apache-2.0", "apache-2.0", "LiheYoung/depth-anything-small-hf"),
  "onnx-community/dfine_n_coco-ONNX": open("Apache-2.0", "apache-2.0", "ustc-community/dfine-nano-coco"),
  "onnx-community/dfine_s_coco-ONNX": open("Apache-2.0", "apache-2.0", "ustc-community/dfine-small-coco"),
  "Xenova/yolos-small": open("Apache-2.0", "apache-2.0", "hustvl/yolos-small"),
  "onnx-community/rtdetr_r50vd": open("Apache-2.0", "apache-2.0", "PekingU/rtdetr_r50vd"),
  "Xenova/detr-resnet-50": open("Apache-2.0", "apache-2.0", "facebook/detr-resnet-50"),
  "Xenova/detr-resnet-50-panoptic": open("Apache-2.0", "apache-2.0", "facebook/detr-resnet-50-panoptic"),
  "Xenova/segformer-b0-finetuned-ade-512-512": {
    name: "NVIDIA Source Code License (SegFormer)",
    terms: "non-commercial",
    url: SEGFORMER_LICENCE,
    from: "nvidia/segformer-b0-finetuned-ade-512-512",
    hub: "other",
    note: "NVIDIA licenses SegFormer for research or evaluation only.",
  },
  "mattmdjaga/segformer_b2_clothes": {
    name: "NVIDIA Source Code License (SegFormer)",
    terms: "non-commercial",
    url: SEGFORMER_LICENCE,
    hub: "other",
    note: "A SegFormer fine-tune, so NVIDIA's research-or-evaluation-only licence carries over.",
  },
  "Xenova/face-parsing": {
    name: "Non-commercial research and education",
    terms: "non-commercial",
    url: hf("jonathandinu/face-parsing"),
    from: "jonathandinu/face-parsing",
    hub: null,
    note: "Its model card limits use to non-commercial research and educational purposes.",
  },
  "Xenova/clip-vit-base-patch32": {
    name: "MIT",
    terms: "permissive",
    url: "https://github.com/openai/CLIP/blob/main/LICENSE",
    from: "openai/clip-vit-base-patch32",
    hub: null,
    via: "OpenAI's CLIP repository licenses the released model under MIT; the model card states no licence.",
  },
  "Xenova/siglip-base-patch16-224": open("Apache-2.0", "apache-2.0", "google/siglip-base-patch16-224"),
  "Xenova/owlv2-base-patch16-ensemble": open("Apache-2.0", "apache-2.0", "google/owlv2-base-patch16-ensemble"),
  "Xenova/owlvit-base-patch32": open("Apache-2.0", "apache-2.0", "google/owlvit-base-patch32"),
  "onnx-community/grounding-dino-tiny-ONNX": own(
    "onnx-community/grounding-dino-tiny-ONNX",
    "Apache-2.0",
    "apache-2.0",
  ),
  "Xenova/dinov2-small": open("Apache-2.0", "apache-2.0", "facebook/dinov2-small"),
  "Xenova/dinov2-base": open("Apache-2.0", "apache-2.0", "facebook/dinov2-base"),
  "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX": {
    name: "DINOv3 License",
    terms: "custom",
    url: "https://ai.meta.com/resources/models-and-libraries/dinov3-license",
    hub: "other",
    note: "Meta's own licence, with use restrictions and redistribution conditions. Read it before building on this model's features.",
  },
  "Xenova/slimsam-77-uniform": own("Xenova/slimsam-77-uniform", "Apache-2.0", "apache-2.0"),
  "onnx-community/sam2.1-hiera-tiny-ONNX": {
    name: "Apache-2.0",
    terms: "permissive",
    url: "https://github.com/facebookresearch/sam2/blob/main/LICENSE",
    hub: null,
    via: "Meta releases the SAM 2 checkpoints under Apache-2.0; this ONNX export's card states no licence.",
  },
  "onnx-community/vitpose-base-simple": {
    name: "Apache-2.0",
    terms: "permissive",
    url: "https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE",
    from: "nielsr/vitpose-base-simple",
    hub: null,
    via: "The ViTPose authors release it under Apache-2.0; the converted checkpoint's card states no licence.",
  },
  "Xenova/modnet": own("Xenova/modnet", "Apache-2.0", "apache-2.0"),
  "briaai/RMBG-1.4": {
    name: "bria-rmbg-1.4",
    terms: "non-commercial",
    url: "https://bria.ai/bria-huggingface-model-license-agreement/",
    hub: "other",
    note: "Released under a Creative Commons licence for non-commercial use. Commercial use needs a separate paid agreement with BRIA.",
  },
  "Xenova/swin2SR-classical-sr-x2-64": open("Apache-2.0", "apache-2.0", "caidas/swin2SR-classical-sr-x2-64"),

  // ── Multimodal ───────────────────────────────────────────────────────────
  "HuggingFaceTB/SmolVLM-256M-Instruct": own("HuggingFaceTB/SmolVLM-256M-Instruct", "Apache-2.0", "apache-2.0"),
  "HuggingFaceTB/SmolVLM-500M-Instruct": own("HuggingFaceTB/SmolVLM-500M-Instruct", "Apache-2.0", "apache-2.0"),
  "HuggingFaceTB/SmolVLM2-256M-Video-Instruct": own(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "Apache-2.0",
    "apache-2.0",
  ),

  // ── Text ─────────────────────────────────────────────────────────────────
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english": open(
    "Apache-2.0",
    "apache-2.0",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  ),
  "Xenova/twitter-roberta-base-sentiment-latest": open(
    "CC-BY-4.0",
    "cc-by-4.0",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
  ),
  "Xenova/finbert": {
    name: "Apache-2.0",
    terms: "permissive",
    url: "https://github.com/ProsusAI/finBERT/blob/master/LICENSE",
    from: "ProsusAI/finbert",
    hub: null,
    via: "Prosus releases FinBERT under Apache-2.0 in its code repository; the model card states no licence.",
  },
  "Xenova/bert-base-NER": own("Xenova/bert-base-NER", "MIT", "mit"),
  "Xenova/bert-base-multilingual-cased-ner-hrl": open(
    "AFL-3.0",
    "afl-3.0",
    "Davlan/bert-base-multilingual-cased-ner-hrl",
  ),
  "Xenova/nli-deberta-v3-xsmall": open("Apache-2.0", "apache-2.0", "cross-encoder/nli-deberta-v3-xsmall"),
  "Xenova/mobilebert-uncased-mnli": {
    name: "No licence stated",
    terms: "unstated",
    url: hf("typeform/mobilebert-uncased-mnli"),
    from: "typeform/mobilebert-uncased-mnli",
    hub: null,
    note: "Its publisher states no licence, so its terms are unknown. Fine to try; do not build on it.",
  },
  "Xenova/distilbert-base-uncased-mnli": {
    name: "No licence stated",
    terms: "unstated",
    url: hf("typeform/distilbert-base-uncased-mnli"),
    from: "typeform/distilbert-base-uncased-mnli",
    hub: null,
    note: "Its publisher lists the licence as unknown. Fine to try; do not build on it.",
  },
  "Xenova/bart-large-mnli": open("MIT", "mit", "facebook/bart-large-mnli"),
  "Xenova/distilbert-base-cased-distilled-squad": open(
    "Apache-2.0",
    "apache-2.0",
    "distilbert/distilbert-base-cased-distilled-squad",
  ),
  "Xenova/bert-base-uncased": open("Apache-2.0", "apache-2.0", "google-bert/bert-base-uncased"),
  "Xenova/distilbert-base-uncased": open("Apache-2.0", "apache-2.0", "distilbert/distilbert-base-uncased"),
  "Xenova/roberta-base": open("MIT", "mit", "FacebookAI/roberta-base"),
  "onnx-community/ModernBERT-base-ONNX": open("Apache-2.0", "apache-2.0", "answerdotai/ModernBERT-base"),
  "Xenova/all-MiniLM-L6-v2": own("Xenova/all-MiniLM-L6-v2", "Apache-2.0", "apache-2.0"),
  "Xenova/bge-base-en-v1.5": own("Xenova/bge-base-en-v1.5", "MIT", "mit"),
  "Xenova/all-mpnet-base-v2": open("Apache-2.0", "apache-2.0", "sentence-transformers/all-mpnet-base-v2"),
  "nomic-ai/nomic-embed-text-v1.5": own("nomic-ai/nomic-embed-text-v1.5", "Apache-2.0", "apache-2.0"),
  "Alibaba-NLP/gte-modernbert-base": own("Alibaba-NLP/gte-modernbert-base", "Apache-2.0", "apache-2.0"),
  "Xenova/opus-mt-en-de": open("CC-BY-4.0", "cc-by-4.0", "Helsinki-NLP/opus-mt-en-de"),
  "Xenova/opus-mt-de-en": open("Apache-2.0", "apache-2.0", "Helsinki-NLP/opus-mt-de-en"),
  "Xenova/opus-mt-en-fr": open("Apache-2.0", "apache-2.0", "Helsinki-NLP/opus-mt-en-fr"),
  "Xenova/opus-mt-fr-en": open("Apache-2.0", "apache-2.0", "Helsinki-NLP/opus-mt-fr-en"),
  "Xenova/opus-mt-en-es": open("Apache-2.0", "apache-2.0", "Helsinki-NLP/opus-mt-en-es"),
  "Xenova/opus-mt-en-zh": open("Apache-2.0", "apache-2.0", "Helsinki-NLP/opus-mt-en-zh"),
  "Xenova/t5-small": open("Apache-2.0", "apache-2.0", "google-t5/t5-small"),
  "Xenova/distilbart-cnn-6-6": own("Xenova/distilbart-cnn-6-6", "Apache-2.0", "apache-2.0"),
  "Xenova/bart-large-cnn": open("MIT", "mit", "facebook/bart-large-cnn"),
  "HuggingFaceTB/SmolLM2-360M-Instruct": own("HuggingFaceTB/SmolLM2-360M-Instruct", "Apache-2.0", "apache-2.0"),
  "Xenova/gpt2": open("MIT", "mit", "openai-community/gpt2"),
  "onnx-community/Qwen2.5-0.5B-Instruct": open("Apache-2.0", "apache-2.0", "Qwen/Qwen2.5-0.5B-Instruct"),
  "Xenova/ms-marco-MiniLM-L-6-v2": open("Apache-2.0", "apache-2.0", "cross-encoder/ms-marco-MiniLM-L6-v2"),
  "mixedbread-ai/mxbai-rerank-xsmall-v1": own("mixedbread-ai/mxbai-rerank-xsmall-v1", "Apache-2.0", "apache-2.0"),
};

/** The halves a composite entry is built from — each one a separate download. */
const HALVES = ["detector", "pose", "embedder", "reranker", "depth"] as const;

/**
 * The Hub repos an entry downloads, in order. A plain entry fetches its own id;
 * a pair (pose, ranking, grounding) fetches its halves; an entry with an
 * explicit `repo` (VAD, enhancement) fetches that, or nothing when it is
 * `null`; and an id that is not a repo at all (`energy`, `behaviour-cloning`)
 * fetches nothing.
 */
export function entryRepos(entry: Pick<CatalogueEntry, "id">): string[] {
  const e = entry as Record<string, unknown>;
  const halves = HALVES.map((k) => e[k])
    .filter((h): h is { id: string } => typeof h === "object" && h !== null && "id" in h)
    .map((h) => h.id);
  if (halves.length) return [...new Set(halves)];
  if ("repo" in e) return typeof e.repo === "string" ? [e.repo] : [];
  return entry.id.includes("/") ? [entry.id] : [];
}

/** The licence of one repo, or `undefined` when the table has no row for it. */
export function licenceFor(repo: string): ModelLicence | undefined {
  return MODEL_LICENCES[repo];
}

/** Worst first: the order a combined entry reports its halves in. */
const SEVERITY: Record<LicenceTerms, number> = {
  "non-commercial": 3,
  custom: 2,
  unstated: 1,
  permissive: 0,
};

export interface RepoLicence {
  repo: string;
  licence: ModelLicence;
}

/**
 * Every licence an entry carries, most restrictive first. A pair is only as
 * usable as its stricter half, so the first element is the one to show when
 * there is room for one line.
 */
export function entryLicences(entry: Pick<CatalogueEntry, "id">): RepoLicence[] {
  return entryRepos(entry)
    .flatMap((repo) => {
      const licence = licenceFor(repo);
      return licence ? [{ repo, licence }] : [];
    })
    .sort((a, b) => SEVERITY[b.licence.terms] - SEVERITY[a.licence.terms]);
}

/** True when the terms need reading before the output is used for anything. */
export function isRestricted(licence: ModelLicence): boolean {
  return licence.terms !== "permissive";
}

/** The two-word label a picker row carries for a restricted licence. */
export const TERMS_LABEL: Record<LicenceTerms, string> = {
  permissive: "Permissive",
  "non-commercial": "Non-commercial",
  custom: "Custom licence",
  unstated: "No licence stated",
};
