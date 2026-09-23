import { describe, expect, it } from "vitest";

import { isHeavyDownload, sizeEstimate } from "@/model/size";

import { entitySlot } from "./highlight";
import {
  CLASSIFIER_SAMPLES,
  DEFAULT_FILL_MASK,
  DEFAULT_NER_MODEL,
  DEFAULT_REDACTED,
  DEFAULT_TEXT_CLASSIFIER,
  DEFAULT_ZERO_SHOT_TEXT,
  FILL_MASK_MODELS,
  MASK_PROBES,
  MASK_SAMPLES,
  MASK_TOKENS,
  NER_MODELS,
  NER_SAMPLES,
  TEXT_CLASSIFIER_MODELS,
  TOP_K,
  ZERO_SHOT_SAMPLES,
  ZERO_SHOT_TEXT_MODELS,
} from "./catalogue";
import { parseLabels } from "./zeroShot";

describe("the text catalogue", () => {
  it("measures every entry's download for both backends", () => {
    // Every task's entries, not just the classifier's: the rule is
    // category-wide, so a new task that forgets it fails here.
    // The category-wide rule, and it is not decoration: #5's size tables were
    // all q8 figures while `loadOpts()` asks for fp16 on WebGPU, which is
    // roughly double all the way down. An entry without both numbers quotes
    // the user a price for a download they are not making.
    for (const m of [
      ...TEXT_CLASSIFIER_MODELS,
      ...NER_MODELS,
      ...ZERO_SHOT_TEXT_MODELS,
      ...FILL_MASK_MODELS,
    ]) {
      expect(m.bytes.webgpu, `${m.id} webgpu bytes`).toBeGreaterThan(0);
      expect(m.bytes.wasm, `${m.id} wasm bytes`).toBeGreaterThan(0);
      // fp16 is two bytes a parameter and q8 is one, so the GPU download is
      // the bigger of the pair. A pair the other way round is a transposed
      // measurement, which reads as plausible and is not.
      expect(m.bytes.webgpu!, `${m.id}`).toBeGreaterThan(m.bytes.wasm!);
    }
  });

  it("warns on the size the user will actually pay, not the smaller one", () => {
    // `sizeEstimate` keys `large` off the bigger of the two downloads. Two of
    // these three cross LARGE_MODEL_BYTES on WebGPU while reading as
    // comfortably under it at q8 — the guardrail only gets that right because
    // the measured bytes are here rather than a params estimate.
    const twitter = TEXT_CLASSIFIER_MODELS.find(
      (m) => m.id === "Xenova/twitter-roberta-base-sentiment-latest",
    )!;
    expect(sizeEstimate(twitter.params, twitter.bytes).large).toBe(true);

    const distil = TEXT_CLASSIFIER_MODELS.find(
      (m) => m.id === DEFAULT_TEXT_CLASSIFIER,
    )!;
    expect(sizeEstimate(distil.params, distil.bytes).large).toBe(false);
  });

  it("offers only checkpoints with a trained classification head", () => {
    // `onnx-community/ModernBERT-base-ONNX` is deliberately absent against the
    // roadmap's §3.1 table: a base encoder has no trained head, so it emits
    // LABEL_0/LABEL_1 from random weights — confident, fluent and meaningless,
    // with nothing failing on the way there. Pinned so it cannot drift back in.
    const ids = TEXT_CLASSIFIER_MODELS.map((m) => m.id);
    expect(ids).not.toContain("onnx-community/ModernBERT-base-ONNX");
    for (const m of TEXT_CLASSIFIER_MODELS) {
      expect(m.labels.length, `${m.id} declares its labels`).toBeGreaterThan(1);
      expect(m.labels.some((l) => /^LABEL_\d+$/.test(l))).toBe(false);
    }
  });

  it("gives every entry a distinct training domain — the page's whole subject", () => {
    const domains = TEXT_CLASSIFIER_MODELS.map((m) => m.domain);
    expect(new Set(domains).size).toBe(domains.length);
  });

  it("shows enough labels to expose a near-tie", () => {
    // TOP_K covers the widest head in the catalogue, so no page ever renders a
    // truncated distribution that looks like a confident one.
    const widest = Math.max(...TEXT_CLASSIFIER_MODELS.map((m) => m.labels.length));
    expect(TOP_K).toBeGreaterThanOrEqual(widest);
  });

  it("defaults to a model in the catalogue", () => {
    expect(TEXT_CLASSIFIER_MODELS.map((m) => m.id)).toContain(
      DEFAULT_TEXT_CLASSIFIER,
    );
  });

  it("ships samples the models disagree on", () => {
    expect(CLASSIFIER_SAMPLES.length).toBeGreaterThanOrEqual(3);
    for (const s of CLASSIFIER_SAMPLES) {
      expect(s.text.trim().length).toBeGreaterThan(0);
      // Each sample says which model it is a trap for — a sample set every
      // model gets right demonstrates nothing about any of them.
      expect(s.hint.trim().length).toBeGreaterThan(0);
    }
    expect(new Set(CLASSIFIER_SAMPLES.map((s) => s.id)).size).toBe(
      CLASSIFIER_SAMPLES.length,
    );
  });
});

describe("the NER catalogue", () => {
  it("ships the multilingual entry the roadmap said was lost", () => {
    // §3.2 sends multilingual NER to a server on the strength of two repos with
    // no ONNX export — but the runtime's own default for `token-classification`
    // is this one, and it is under the feasibility bar. Pinned so the finding
    // is not quietly lost again.
    expect(NER_MODELS.map((m) => m.id)).toContain(
      "Xenova/bert-base-multilingual-cased-ner-hrl",
    );
  });

  it("defaults to a model in the catalogue", () => {
    expect(NER_MODELS.map((m) => m.id)).toContain(DEFAULT_NER_MODEL);
  });

  it("gives every declared entity type a colour slot", () => {
    // An entity the palette has no slot for renders neutral, which is correct
    // but is not what a catalogue entry should be promising by name.
    for (const m of NER_MODELS) {
      for (const type of m.entities) {
        expect(entitySlot(type), `${m.id} → ${type}`).not.toBeNull();
      }
    }
  });

  it("never needs more than the four validated hues at once", () => {
    // Four slots, not five, is the whole reason MISC and DATE share one: only
    // one model is live at a time, so no single page can put five types on
    // screen — and no 5-subset of the palette passes the all-pairs floors.
    for (const m of NER_MODELS) {
      const slots = new Set(m.entities.map(entitySlot));
      expect(slots.size).toBeLessThanOrEqual(4);
    }
  });

  it("redacts the two types that actually identify a person by default", () => {
    expect([...DEFAULT_REDACTED].sort()).toEqual(["LOC", "PER"]);
    // And every default is a type the default model can actually emit, or the
    // button would start with a selection that does nothing.
    const head = NER_MODELS.find((m) => m.id === DEFAULT_NER_MODEL)!;
    for (const type of DEFAULT_REDACTED) {
      expect(head.entities).toContain(type);
    }
  });

  it("ships samples with entities in them", () => {
    expect(NER_SAMPLES.length).toBeGreaterThanOrEqual(3);
    expect(new Set(NER_SAMPLES.map((s) => s.id)).size).toBe(NER_SAMPLES.length);
    for (const s of NER_SAMPLES) {
      expect(s.text.trim().length).toBeGreaterThan(0);
      expect(s.hint.trim().length).toBeGreaterThan(0);
    }
  });
});

describe("the zero-shot catalogue", () => {
  it("keeps the page's floor small, which is what the feasibility bar asks", () => {
    // §0's bar is about the **floor**, not the ceiling: a cheap default beside
    // a gated heavy option is fine, and a page whose every entry is heavy is
    // the failure — that is what removed /text-to-audio, /image-to-text and
    // /document-question-answering. This page's floor is MobileBERT at 26 MB
    // on WASM, so BART-large is an option rather than the price of entry.
    const floor = Math.min(
      ...ZERO_SHOT_TEXT_MODELS.map((m) =>
        Math.min(m.bytes.webgpu ?? Infinity, m.bytes.wasm ?? Infinity),
      ),
    );
    expect(floor).toBeLessThan(30 * 1024 * 1024);
  });

  it("gates BART-large, and nothing else, behind the second opt-in", () => {
    // 816 MB on WebGPU — and the roadmap's "411 MB, under the bar" was its q8
    // size, which is not the number `loadOpts()` asks for on a machine with an
    // adapter. The gate keys off the measured bytes, so quoting the smaller
    // figure would silently disarm it.
    const gated = ZERO_SHOT_TEXT_MODELS.filter((m) => isHeavyDownload(m.bytes));
    expect(gated.map((m) => m.id)).toEqual(["Xenova/bart-large-mnli"]);
    // ...and it stays inside the browser budget the guide sets (~1 GB), which
    // is the line Qwen3-VL-2B fell the wrong side of.
    for (const m of ZERO_SHOT_TEXT_MODELS) {
      expect(Math.max(m.bytes.webgpu ?? 0, m.bytes.wasm ?? 0), m.id).toBeLessThan(
        1024 ** 3,
      );
    }
  });

  it("does not offer the checkpoint the roadmap recommended as its default", () => {
    // `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` publishes one fp32
    // `onnx/model.onnx` at 738.6 MB — the "~180 MB" #5 quoted was an estimate
    // of a file that does not exist. Pinned so it cannot drift back in off the
    // strength of that table.
    expect(ZERO_SHOT_TEXT_MODELS.map((m) => m.id)).not.toContain(
      "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    );
  });

  it("defaults to a model in the catalogue, and to the cheap one", () => {
    const ids = ZERO_SHOT_TEXT_MODELS.map((m) => m.id);
    expect(ids).toContain(DEFAULT_ZERO_SHOT_TEXT);
    const fallback = ZERO_SHOT_TEXT_MODELS.find(
      (m) => m.id === DEFAULT_ZERO_SHOT_TEXT,
    )!;
    expect(isHeavyDownload(fallback.bytes)).toBe(false);
  });

  it("ships samples that bring their own labels", () => {
    // Half a zero-shot example is not an example: the point of the task is
    // that the label set is the user's, so a sample that fills only the text
    // box demonstrates nothing /text-classification did not.
    expect(ZERO_SHOT_SAMPLES.length).toBeGreaterThanOrEqual(3);
    for (const s of ZERO_SHOT_SAMPLES) {
      expect(s.text.trim().length, s.id).toBeGreaterThan(0);
      expect(s.hint.trim().length, s.id).toBeGreaterThan(0);
      expect(s.labels.length, `${s.id} labels`).toBeGreaterThan(1);
      // Bare nouns, so the template composes: a label carrying its own article
      // yields "This example is a a billing issue."
      for (const label of s.labels) {
        expect(label, `${s.id}: ${label}`).not.toMatch(/^(a|an|the) /i);
      }
      // And the set survives the editor's own parser unchanged — a sample with
      // a comma in a label would silently split into two.
      expect(parseLabels(s.labels.join("\n"))).toEqual([...s.labels]);
    }
    expect(new Set(ZERO_SHOT_SAMPLES.map((s) => s.id)).size).toBe(
      ZERO_SHOT_SAMPLES.length,
    );
  });
});

describe("the fill-mask catalogue", () => {
  it("declares a mask token for every entry", () => {
    // Declared as data so the page can show it — and insert it — before the
    // 219 MB download that would otherwise be the only way to find out.
    for (const m of FILL_MASK_MODELS) {
      expect(m.maskToken, `${m.id} mask token`).toBeTruthy();
      expect(m.task).toBe("fill-mask");
    }
  });

  it("ships more than one tokenizer family, which is the page's whole point", () => {
    // A catalogue where every entry says `[MASK]` cannot demonstrate the hazard
    // the page exists to prevent, and the bug would then only appear for a user
    // who pasted a foreign token. Measured on the Hub: three of these four use
    // `[MASK]` and `Xenova/roberta-base` uses `<mask>`.
    expect(MASK_TOKENS.length).toBeGreaterThan(1);
    expect(MASK_TOKENS).toContain("[MASK]");
    expect(MASK_TOKENS).toContain("<mask>");
  });

  it("derives the known-token list from the entries, with no duplicates", () => {
    // `retargetMasks` rewrites *from* this list, so a fifth family must arrive
    // as a catalogue entry rather than as an edit to `text/mask.ts`.
    expect([...new Set(MASK_TOKENS)]).toEqual([...MASK_TOKENS]);
    for (const m of FILL_MASK_MODELS) {
      expect(MASK_TOKENS).toContain(m.maskToken);
    }
  });

  it("defaults to a model in its own list", () => {
    expect(FILL_MASK_MODELS.map((m) => m.id)).toContain(DEFAULT_FILL_MASK);
  });

  it("keeps a literal mask token out of every sample and probe", () => {
    // The samples are templates precisely so they cannot carry one model's
    // token into another model's box — which is the page's own bug, shipped as
    // content.
    for (const s of MASK_SAMPLES) {
      expect(s.template, s.id).toContain("{}");
      for (const token of MASK_TOKENS) {
        expect(s.template, s.id).not.toContain(token);
      }
    }
    for (const p of MASK_PROBES) {
      for (const t of p.templates) {
        expect(t, p.id).toContain("{}");
        for (const token of MASK_TOKENS) expect(t, p.id).not.toContain(token);
      }
    }
  });

  it("gives every probe a pair that differs by exactly one word", () => {
    // The pair is the measurement. Two prompts that differ in more than one
    // place cannot attribute the difference in the answers to anything.
    for (const p of MASK_PROBES) {
      const [a, b] = p.templates.map((t) => t.split(/\s+/));
      expect(a.length, `${p.id} lengths`).toBe(b.length);
      const differing = a.filter((w, i) => w !== b[i]);
      expect(differing, `${p.id} differing words`).toHaveLength(1);
      // And the word that differs is the one the columns are labelled with.
      expect(p.varies.some((v) => differing[0].includes(v))).toBe(true);
    }
  });

  it("labels each probe column with the word that varies", () => {
    for (const p of MASK_PROBES) {
      expect(p.varies[0]).not.toBe(p.varies[1]);
    }
  });
});
