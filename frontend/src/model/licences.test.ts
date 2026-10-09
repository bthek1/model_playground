import { describe, expect, it } from "vitest";

import { MODEL_CATALOGUES } from "@/legal/catalogues";
import { POSE_MODELS } from "@/vision/pose/types";
import { RANKING_PAIRS } from "@/text/catalogue";
import { GROUNDING_PAIR, ROBOTICS_ENTRIES } from "@/vision/grounding";
import { VAD_MODELS } from "@/audio/vad/types";

import {
  MODEL_LICENCES,
  entryLicences,
  entryRepos,
  isRestricted,
  licenceFor,
} from "./licences";

/** Every repo any page can download, with the catalogue that offers it. */
const downloaded = MODEL_CATALOGUES.flatMap((c) =>
  c.entries.flatMap((e) => entryRepos(e).map((repo) => ({ repo, catalogue: c.name }))),
);

describe("MODEL_LICENCES", () => {
  it("covers every repo every catalogue downloads", () => {
    // The table's whole job. A model with no row reaches a page with no
    // licence line and no row on /licences — the attribution the licence
    // requires, silently missing.
    const missing = downloaded.filter((d) => !licenceFor(d.repo)).map((d) => `${d.catalogue}: ${d.repo}`);
    expect(missing).toEqual([]);
  });

  it("has no row for a repo nothing downloads", () => {
    // A stale row is how a cut model's licence stays on /licences forever.
    const used = new Set(downloaded.map((d) => d.repo));
    expect(Object.keys(MODEL_LICENCES).filter((r) => !used.has(r))).toEqual([]);
  });

  it("links every licence over https", () => {
    for (const [repo, l] of Object.entries(MODEL_LICENCES)) {
      expect(l.url, repo).toMatch(/^https:\/\/\S+$/);
      expect(l.name, repo).toBeTruthy();
    }
  });

  it("explains every restriction in a sentence", () => {
    // "bria-rmbg-1.4" or "other" tells a reader nothing; the note is what
    // they can act on.
    for (const [repo, l] of Object.entries(MODEL_LICENCES)) {
      if (isRestricted(l)) expect(l.note, repo).toMatch(/\.\s*$/);
    }
  });

  it("says where a licence comes from when the model card is silent", () => {
    // `hub: null` with permissive terms is a claim made on some other
    // authority, so it has to name that authority.
    for (const [repo, l] of Object.entries(MODEL_LICENCES)) {
      if (l.hub === null && l.terms === "permissive") expect(l.via, repo).toBeTruthy();
    }
  });

  it("never names the downloaded repo as its own upstream", () => {
    for (const [repo, l] of Object.entries(MODEL_LICENCES)) {
      expect(l.from, repo).not.toBe(repo);
    }
  });
});

describe("entryRepos", () => {
  it("is the entry's own id for a plain checkpoint", () => {
    expect(entryRepos({ id: "Xenova/modnet" })).toEqual(["Xenova/modnet"]);
  });

  it("is both halves of a pair, never the composite id", () => {
    for (const pair of [...POSE_MODELS, ...RANKING_PAIRS, GROUNDING_PAIR]) {
      const repos = entryRepos(pair);
      expect(repos.length, pair.id).toBe(2);
      expect(repos, pair.id).not.toContain(pair.id);
    }
  });

  it("is nothing for an entry that downloads nothing", () => {
    const energy = VAD_MODELS.find((m) => m.repo === null)!;
    expect(entryRepos(energy)).toEqual([]);
    const cloning = ROBOTICS_ENTRIES.find((e) => e.kind === "cloning")!;
    expect(entryRepos(cloning)).toEqual([]);
  });

  it("is the explicit repo when an entry names one", () => {
    const silero = VAD_MODELS.find((m) => m.repo !== null)!;
    expect(entryRepos(silero)).toEqual([silero.repo]);
  });
});

describe("entryLicences", () => {
  it("puts the stricter half of a pair first", () => {
    // A pair is only as usable as its more restrictive half, and the picker
    // badge shows the first element.
    const pair = {
      id: "x+y",
      detector: { id: "Xenova/modnet" },
      pose: { id: "briaai/RMBG-1.4" },
    };
    expect(entryLicences(pair).map((l) => l.repo)).toEqual(["briaai/RMBG-1.4", "Xenova/modnet"]);
  });

  it("names the restricted models the audit found", () => {
    // Pinned so a catalogue edit that swaps one of these out — or a row
    // quietly relabelled permissive — is a visible diff in this test.
    const restricted = Object.entries(MODEL_LICENCES)
      .filter(([, l]) => isRestricted(l))
      .map(([repo, l]) => `${repo}: ${l.terms}`)
      .sort();
    expect(restricted).toEqual([
      "Xenova/distilbert-base-uncased-mnli: unstated",
      "Xenova/face-parsing: non-commercial",
      "Xenova/mms-tts-eng: non-commercial",
      "Xenova/mobilebert-uncased-mnli: unstated",
      "Xenova/segformer-b0-finetuned-ade-512-512: non-commercial",
      "briaai/RMBG-1.4: non-commercial",
      "mattmdjaga/segformer_b2_clothes: non-commercial",
      "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX: custom",
    ]);
  });
});
