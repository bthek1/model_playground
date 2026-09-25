import { describe, expect, it } from "vitest";

import { SEQ2SEQ_WASM_DTYPES } from "@/model/backend";
import { HEAVY_MODEL_BYTES, isHeavyDownload, sizeEstimate } from "@/model/size";

import { entitySlot } from "./highlight";
import { lead, sentenceCount, wordCount } from "./lead3";
import {
  ARTICLE_SAMPLES,
  CLASSIFIER_SAMPLES,
  DEFAULT_EMBED_MODEL,
  DEFAULT_FILL_MASK,
  DEFAULT_NER_MODEL,
  DEFAULT_REDACTED,
  DECODING_PRESETS,
  DEFAULT_SUMMARIZER,
  DEFAULT_TEXTGEN_MODEL,
  DEFAULT_TEXT_CLASSIFIER,
  DEFAULT_TRANSLATION_MODEL,
  DEFAULT_ZERO_SHOT_TEXT,
  EMBED_MODELS,
  FAITHFULNESS_MODEL,
  FEATURE_TEXT_SAMPLES,
  FILL_MASK_MODELS,
  MASK_PROBES,
  MASK_SAMPLES,
  MASK_TOKENS,
  NER_MODELS,
  NER_SAMPLES,
  LEAD_N,
  NLLB_BYTES,
  PAIR_SAMPLES,
  SUMMARIZER_MODELS,
  TEXTGEN_MODELS,
  TEXTGEN_SAMPLES,
  TEXT_CLASSIFIER_MODELS,
  TOP_K,
  TRANSLATION_MODELS,
  TRANSLATION_SAMPLES,
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
      ...EMBED_MODELS,
      ...TRANSLATION_MODELS,
      ...SUMMARIZER_MODELS,
    ]) {
      // Both numbers, **for the backends the entry claims**. Two summarization
      // entries have no CPU path at all — a quantized seq2seq decoder will not
      // open a WASM session and the unquantized build is past the size bar — so
      // they declare `backends: ["webgpu"]`, and quoting a WASM figure for a
      // download that is never offered would be a price for a thing that is not
      // for sale. `model-ids.spec.ts` scopes its Hub check the same way.
      for (const backend of m.backends ?? (["webgpu", "wasm"] as const)) {
        expect(m.bytes[backend], `${m.id} ${backend} bytes`).toBeGreaterThan(0);
      }
      // fp16 is two bytes a parameter and q8 is one, so the GPU download is
      // normally the bigger of the pair, and a transposed measurement reads as
      // plausible. **Unless the entry pins a more expensive WASM precision** —
      // which every seq2seq entry does, because a quantized decoder cannot
      // open a session on the bundled WASM provider at all
      // (`SEQ2SEQ_WASM_DTYPES`). So the invariant is not "webgpu is bigger", it
      // is "an inversion is *explained by a pin*, never accidental".
      if ((m.bytes.wasm ?? 0) > (m.bytes.webgpu ?? 0)) {
        expect(
          m.dtypes?.wasm,
          `${m.id} quotes a bigger WASM download than WebGPU without pinning a WASM dtype — that is a transposed measurement, not a precision decision`,
        ).toBeDefined();
      }
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

describe("the embedding catalogue", () => {
  // The cheapest floor in the app, and the reason §3.7's two pages exist at
  // all: the default download is 22 MiB. `Qwen3-Embedding-0.6B` measures
  // 613.5 MB and is cut, as the roadmap says.
  it("keeps the floor tiny, which is the whole argument for the page", () => {
    const smallest = Math.min(
      ...EMBED_MODELS.map((m) => Math.min(m.bytes.webgpu!, m.bytes.wasm!)),
    );
    expect(smallest).toBeLessThan(30e6);
    expect(
      EMBED_MODELS.some((m) => m.id.includes("Qwen3-Embedding")),
      "Qwen3-Embedding is 613.5 MB and cut per roadmap §3.7",
    ).toBe(false);
  });

  // The field nothing downstream can discover, because the `Xenova/*` ONNX
  // mirrors do not carry `1_Pooling/config.json`. Getting it wrong returns a
  // vector of the right width that ranks plausibly and is wrong, with nothing
  // failing — so every entry declares it, and names the repo that is the
  // authority for it (`just fe-e2e-models` checks the pair).
  it("declares a pooling and an upstream repo for every entry", () => {
    for (const m of EMBED_MODELS) {
      expect(["mean", "cls"], `${m.id} pooling`).toContain(m.pooling);
      expect(m.upstream, `${m.id} upstream`).toMatch(/^[^/]+\/[^/]+$/);
      expect(m.dim, `${m.id} dim`).toBeGreaterThan(0);
    }
  });

  // Both poolings one click apart, deliberately: a catalogue where every entry
  // is mean-pooled makes the field look like decoration.
  it("ships both poolings, so the hazard is reachable rather than theoretical", () => {
    const poolings = new Set(EMBED_MODELS.map((m) => m.pooling));
    expect(poolings).toEqual(new Set(["mean", "cls"]));
  });

  // The truncation control means two different things depending on this flag —
  // a demonstration on an MRL checkpoint and a *measurement* on the others — so
  // it cannot be assumed, and the plan's four original entries are all
  // non-MRL.
  it("marks exactly the Matryoshka-trained entry, and has one", () => {
    const mrl = EMBED_MODELS.filter((m) => m.matryoshka);
    expect(mrl).toHaveLength(1);
    expect(mrl[0].id).toBe("nomic-ai/nomic-embed-text-v1.5");
    // And that entry is the one that needs a task prefix, which is why the
    // prefix machinery is not dead code.
    expect(mrl[0].prefixes).toBeDefined();
  });

  it("names a prefix per use, so a page cannot pick the wrong one", () => {
    for (const m of EMBED_MODELS) {
      if (!m.prefixes) continue;
      for (const kind of ["symmetric", "query", "document"] as const) {
        expect(m.prefixes[kind], `${m.id} ${kind}`).toBeTruthy();
      }
      // Asymmetric retrieval needs the two halves to differ, or the prefix is
      // doing nothing on `/text-ranking`.
      expect(m.prefixes.query).not.toBe(m.prefixes.document);
    }
  });

  it("defaults to a model in its own list, and to the cheap one", () => {
    const def = EMBED_MODELS.find((m) => m.id === DEFAULT_EMBED_MODEL);
    expect(def).toBeDefined();
    expect(def!.bytes.wasm!).toBeLessThan(30e6);
  });

  it("ships pair samples that span the range, including where the models fail", () => {
    expect(PAIR_SAMPLES.length).toBeGreaterThanOrEqual(3);
    for (const p of PAIR_SAMPLES) {
      expect(p.a.length).toBeGreaterThan(0);
      expect(p.b.length).toBeGreaterThan(0);
      expect(p.a).not.toBe(p.b);
      expect(p.hint.length).toBeGreaterThan(0);
    }
    // The negation pair is the honest one and ships with its own caveat: these
    // models score a sentence and its negation as very similar, which is a
    // limitation of embeddings rather than a bug in the page.
    const negation = PAIR_SAMPLES.find((p) => p.id === "negation");
    expect(negation, "the negation pair").toBeDefined();
    expect(negation!.hint).toMatch(/limitation/i);

    // And a floor to measure the others against — a page with no unrelated
    // pair has no scale.
    expect(PAIR_SAMPLES.some((p) => p.id === "unrelated")).toBe(true);
  });

  it("ships short feature samples rather than walls of prose", () => {
    expect(FEATURE_TEXT_SAMPLES.length).toBeGreaterThan(0);
    for (const s of FEATURE_TEXT_SAMPLES) {
      expect(s.text.length).toBeGreaterThan(0);
      expect(s.text.length).toBeLessThan(200);
      expect(s.hint.length).toBeGreaterThan(0);
    }
  });
});

describe("the translation catalogue", () => {
  it("gives every entry a distinct direction, labelled rather than repo-named", () => {
    const directions = TRANSLATION_MODELS.map((m) => `${m.source}-${m.target}`);
    expect(new Set(directions).size).toBe(TRANSLATION_MODELS.length);
    for (const m of TRANSLATION_MODELS) {
      // The row's name is the direction, not `Xenova/opus-mt-en-de`.
      expect(m.label).toContain("→");
      expect(m.direction).toBe(m.label);
      expect(m.source).not.toBe(m.target);
    }
  });

  // Six rather than two, so "a specialist beats one generalist" is visible: the
  // whole catalogue is still comfortably more useful than one NLLB download,
  // and each *individual* pair is a fraction of it.
  it("ships enough pairs to make the specialists argument, and stays under NLLB per pair", () => {
    expect(TRANSLATION_MODELS.length).toBeGreaterThanOrEqual(6);
    for (const m of TRANSLATION_MODELS) {
      expect(
        Math.max(m.bytes.webgpu!, m.bytes.wasm!),
        `${m.id} against one NLLB download`,
      ).toBeLessThan(NLLB_BYTES / 2);
    }
  });

  // A measurement, not a precaution: a Marian decoder's session does not open
  // at all on the WASM provider bundled with 4.2.0. Every entry references the
  // one spec in `model/backend.ts` rather than repeating the literal.
  it("pins the WASM precision on every pair, via the shared seq2seq spec", () => {
    for (const m of TRANSLATION_MODELS) {
      expect(m.dtypes?.wasm, `${m.id} wasm dtype`).toEqual(SEQ2SEQ_WASM_DTYPES);
      // And declares the two graphs it actually downloads, so the Hub check
      // looks at the right files — `model.onnx` does not exist in these repos.
      expect(m.graphs).toEqual(["encoder_model", "decoder_model_merged"]);
    }
  });

  // The open question the plan left for Phase 2, and it resolves itself. The
  // plan worried en↔de (199.6 MiB at fp16) would slip under LARGE_MODEL_BYTES
  // while en→es (213.1 MiB) crossed it, leaving one warning on a page of
  // identical models. With the WASM pin the CPU download is 271–289 MB for
  // every pair and `large` keys off the bigger of the two, so every pair warns
  // — consistently, about a number the user will actually pay.
  it("warns consistently across every pair, not on some of them", () => {
    const warns = TRANSLATION_MODELS.map(
      (m) => sizeEstimate(m.params, m.bytes).large,
    );
    expect(warns.every(Boolean), "every pair warns").toBe(true);
    // And none is heavy enough for the second opt-in — these are ~200-290 MB.
    for (const m of TRANSLATION_MODELS) {
      expect(isHeavyDownload(m.bytes), `${m.id}`).toBe(false);
    }
  });

  it("defaults to a pair in its own list", () => {
    expect(
      TRANSLATION_MODELS.some((m) => m.id === DEFAULT_TRANSLATION_MODEL),
    ).toBe(true);
  });

  // A pair with no sample text is a page that cannot be driven without typing
  // in a language the user may not speak.
  it("ships samples for every source language it offers", () => {
    for (const m of TRANSLATION_MODELS) {
      const samples = TRANSLATION_SAMPLES[m.source];
      expect(samples, `samples for ${m.source}`).toBeDefined();
      expect(samples.length).toBeGreaterThan(0);
      for (const s of samples) {
        expect(s.text.length).toBeGreaterThan(0);
        expect(s.hint.length).toBeGreaterThan(0);
      }
    }
  });
});

describe("the summarization catalogue", () => {
  // The whole page turns on one measurement: distilbart opens a q8 session on
  // WebGPU. Every other configuration is over §0's bar, so the pin is not a
  // preference and un-pinning it would quietly take the page over the bar.
  it("pins q8 on WebGPU for the BART entries, which is what keeps them in budget", () => {
    for (const m of SUMMARIZER_MODELS) {
      if (!m.id.includes("bart")) continue;
      expect(m.dtypes?.webgpu, `${m.id} webgpu dtype`).toBe("q8");
      expect(
        m.bytes.webgpu!,
        `${m.id} must stay inside the ~500 MB feasibility bar`,
      ).toBeLessThan(HEAVY_MODEL_BYTES);
    }
  });

  // A quantized seq2seq decoder cannot open a WASM session at all, and unlike
  // Marian the fp32-decoder fallback does not fit here (742.8 MB). So the BART
  // entries have no CPU path and must say so, or the picker offers a download
  // that fails at the end of itself.
  it("declares the BART entries GPU-only rather than letting the load fail", () => {
    for (const m of SUMMARIZER_MODELS) {
      if (!m.id.includes("bart")) continue;
      expect(m.backends, `${m.id} backends`).toEqual(["webgpu"]);
      expect(m.bytes.wasm, `${m.id} must not quote a WASM price`).toBeUndefined();
    }
  });

  // Which would leave the page with no floor at all — so the default is the
  // entry that does run on both, and it is the cheap one.
  it("keeps a floor that runs on CPU, and defaults to it", () => {
    const def = SUMMARIZER_MODELS.find((m) => m.id === DEFAULT_SUMMARIZER);
    expect(def, "the default is in its own list").toBeDefined();
    expect(def!.backends, "the default must run everywhere").toBeUndefined();
    expect(def!.bytes.wasm!).toBeLessThan(HEAVY_MODEL_BYTES);
    expect(def!.dtypes?.wasm, "…which needs the seq2seq pin").toEqual(
      SEQ2SEQ_WASM_DTYPES,
    );
  });

  it("declares the two graphs every seq2seq entry downloads", () => {
    for (const m of SUMMARIZER_MODELS) {
      expect(m.graphs).toEqual(["encoder_model", "decoder_model_merged"]);
    }
  });

  // The page's subject is the baseline, so the samples have to be ones where the
  // baseline is *hard* — news, written answer-first. A sample set of rambling
  // prose would flatter the model.
  it("ships articles long enough for lead-3 to be a real baseline", () => {
    expect(ARTICLE_SAMPLES.length).toBeGreaterThanOrEqual(3);
    for (const s of ARTICLE_SAMPLES) {
      expect(sentenceCount(s.text), `${s.id} sentences`).toBeGreaterThan(LEAD_N);
      expect(wordCount(s.text), `${s.id} words`).toBeGreaterThan(60);
      expect(s.hint.length).toBeGreaterThan(0);
      // And lead-3 is a strict prefix of the article, never the whole thing —
      // otherwise the comparison is against the input itself.
      const baseline = lead(s.text, LEAD_N);
      expect(s.text).toContain(baseline);
      expect(baseline.length).toBeLessThan(s.text.length);
    }
  });

  // The faithfulness model is borrowed rather than added: it is the zero-shot
  // page's cheapest entry, so the second opt-in costs tens of megabytes rather
  // than hundreds.
  it("borrows its entailment model from the zero-shot catalogue", () => {
    const borrowed = ZERO_SHOT_TEXT_MODELS.find(
      (m) => m.id === FAITHFULNESS_MODEL,
    );
    expect(borrowed, "the faithfulness model is a zero-shot entry").toBeDefined();
    expect(borrowed!.task).toBe("zero-shot-classification");
    // The cheapest one, and by a distance — a second model is only honest here
    // if it is small.
    const cheapest = Math.min(
      ...ZERO_SHOT_TEXT_MODELS.map((m) => m.bytes.wasm!),
    );
    expect(borrowed!.bytes.wasm!).toBe(cheapest);
    expect(borrowed!.bytes.wasm!).toBeLessThan(50e6);
  });
});

describe("the text-generation catalogue", () => {
  // The roadmap wanted GPT-2 as the default because it loops so readily. The
  // demonstration is worth keeping; the default is not, now that GPT-2 measures
  // 251 MB with no CPU path at all while SmolLM2 measures 273 MB genuinely
  // quantized and runs on both.
  it("defaults to the entry that has a CPU path", () => {
    const def = TEXTGEN_MODELS.find((m) => m.id === DEFAULT_TEXTGEN_MODEL);
    expect(def, "the default is in its own list").toBeDefined();
    expect(def!.backends, "the default must run everywhere").toBeUndefined();
    expect(def!.bytes.wasm, "…so it needs a measured WASM size").toBeGreaterThan(0);
    // And it is instruction-tuned, so the page's default prompt is answered
    // rather than continued.
    expect(def!.instruct).toBe(true);
  });

  // GPT-2's three measured problems, pinned so the entry cannot drift back to a
  // configuration that does not exist: `model_q4f16.onnx` is the same size as
  // fp16 and not actually quantized, `model_quantized.onnx` is absent so a
  // default `q8` 404s, and the 128.3 MB legacy graph fails to open a session
  // (`transformer.wte.weight_merged_0_scale`).
  it("keeps GPT-2 at fp16 on WebGPU only, which is its one loadable build", () => {
    const gpt2 = TEXTGEN_MODELS.find((m) => m.id === "Xenova/gpt2");
    expect(gpt2, "GPT-2 is kept for the loop demonstration").toBeDefined();
    expect(gpt2!.dtypes?.webgpu).toBe("fp16");
    expect(gpt2!.backends).toEqual(["webgpu"]);
    expect(gpt2!.bytes.wasm, "there is no CPU path to quote").toBeUndefined();
    // fp16 weights, so the adapter feature is a hard requirement rather than a
    // preference — without it the download succeeds and every run fails.
    expect(gpt2!.requireShaderF16).toBe(true);
    // Not instruction-tuned: it continues text, it does not answer.
    expect(gpt2!.instruct).toBe(false);
  });

  // #30's finding, applied as an invariant rather than remembered per entry: an
  // f16 build without the adapter feature loads, reports ready, and fails on the
  // first operator of every run — after the download is paid for.
  it("requires shader-f16 on every entry that loads f16 weights", () => {
    for (const m of TEXTGEN_MODELS) {
      const f16 =
        m.dtypes?.webgpu === "fp16" ||
        m.dtypes?.webgpu === "q4f16" ||
        // No pin means the family default, which is q4f16.
        m.dtypes?.webgpu === undefined;
      if (!f16) continue;
      // The default entry is the exception and states why: it has a working
      // WASM fallback, so its row stays enabled and the *worker* re-asks the
      // question through `pickBackendForF16`.
      if (!m.backends || m.backends.includes("wasm")) continue;
      expect(m.requireShaderF16, `${m.id} loads f16 weights`).toBe(true);
    }
  });

  it("measures the download for every backend each entry claims", () => {
    for (const m of TEXTGEN_MODELS) {
      for (const backend of m.backends ?? (["webgpu", "wasm"] as const)) {
        expect(m.bytes[backend], `${m.id} ${backend}`).toBeGreaterThan(0);
      }
      // And stays inside the feasibility bar — this is a category of models
      // where the next rung up is a gigabyte.
      expect(
        Math.max(m.bytes.webgpu ?? 0, m.bytes.wasm ?? 0),
        `${m.id} against the ~500 MB bar`,
      ).toBeLessThan(HEAVY_MODEL_BYTES);
    }
  });

  describe("the decoding presets", () => {
    // Greedy first and greedy by default: it is reproducible, which is the only
    // thing that makes a repetition loop attributable to the *strategy* rather
    // than blamed on the weights.
    it("opens with a reproducible one", () => {
      expect(DECODING_PRESETS.length).toBeGreaterThanOrEqual(3);
      expect(DECODING_PRESETS[0].decoding.doSample).toBe(false);
    });

    it("ships both regimes, so the comparison has something to compare", () => {
      expect(DECODING_PRESETS.some((p) => p.decoding.doSample)).toBe(true);
      expect(DECODING_PRESETS.some((p) => !p.decoding.doSample)).toBe(true);
    });

    // The page's demonstration: greedy with no repetition penalty is where a
    // small decoder degenerates. A preset set that quietly penalised repetition
    // everywhere would hide the thing the page is about.
    it("includes a preset with the repetition penalty off", () => {
      const loop = DECODING_PRESETS.find((p) => p.id === "loop");
      expect(loop).toBeDefined();
      expect(loop!.decoding.repetitionPenalty).toBe(1.0);
      expect(loop!.decoding.doSample).toBe(false);
    });

    it("gives every preset a distinct id and a reason", () => {
      expect(new Set(DECODING_PRESETS.map((p) => p.id)).size).toBe(
        DECODING_PRESETS.length,
      );
      for (const p of DECODING_PRESETS) {
        expect(p.hint.length).toBeGreaterThan(0);
        expect(p.decoding.maxNewTokens).toBeGreaterThan(0);
      }
    });
  });

  // The `@slow` spec asserts a known continuation, so one sample has to have
  // one — a page whose every prompt is open-ended cannot be checked at all.
  it("ships a prompt with exactly one right continuation", () => {
    const anchor = TEXTGEN_SAMPLES.find((s) => s.id === "unambiguous");
    expect(anchor).toBeDefined();
    expect(anchor!.text).toMatch(/capital of France/i);
    for (const s of TEXTGEN_SAMPLES) {
      expect(s.text.length).toBeGreaterThan(0);
      expect(s.hint.length).toBeGreaterThan(0);
    }
  });
});
