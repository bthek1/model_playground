// Every dataset and sample collection the app bundles or fetches (#62),
// with what its publisher says about reuse.
//
// The tabular and forecasting samples already carry their licence in their
// own catalogues (`tabular/samples.ts`, `forecast/samples.ts`) and render it
// where they are offered; `/licences` lists those too, straight from those
// modules, so this file holds only the datasets that had no licence line
// anywhere.
//
// Several say "no licence stated". That is recorded as found rather than
// smoothed over: they are standard research benchmarks redistributed by every
// ML library, used here for teaching, and the honest description of their
// terms is that their publishers never set any.

import type { LicenceTerms } from "@/model/licences";

export interface DatasetCredit {
  id: string;
  name: string;
  /** What it is, and who made it. */
  credit: string;
  licence: string;
  terms: LicenceTerms;
  /** Where the licence (or its absence) is stated. */
  url: string;
  /** Where the browser gets it from. */
  host: "bundled" | "huggingface" | "google";
  pages: readonly string[];
}

export const DATASETS: readonly DatasetCredit[] = [
  {
    id: "cora",
    name: "Cora citation network",
    credit:
      "McCallum et al. (2000), from the LINQS release. Re-encoded by scripts/prepare-cora.mjs.",
    licence: "No licence stated by its publishers; distributed for research use",
    terms: "unstated",
    url: "https://linqs.org/datasets/#cora",
    host: "bundled",
    pages: ["/graph", "/link-prediction"],
  },
  {
    id: "proteins",
    name: "PROTEINS",
    credit:
      "Borgwardt et al. (2005), from the TUDataset collection (Morris et al., 2020), via graphs-datasets/PROTEINS.",
    licence: "Listed as “unknown” by its Hub card",
    terms: "unstated",
    url: "https://huggingface.co/datasets/graphs-datasets/PROTEINS",
    host: "huggingface",
    pages: ["/graph-classification"],
  },
  {
    id: "mnist",
    name: "MNIST",
    credit: "LeCun, Cortes & Burges. The sprite sheet is the TensorFlow.js tutorial's copy.",
    licence: "CC BY-SA 3.0",
    terms: "permissive",
    url: "https://creativecommons.org/licenses/by-sa/3.0/",
    host: "google",
    pages: ["/training"],
  },
  {
    id: "karate",
    name: "Zachary's karate club",
    credit:
      "W. W. Zachary, “An information flow model for conflict and fission in small groups” (1977). Edge list as published in networkx (BSD-3-Clause).",
    licence: "Facts from a published paper; networkx's copy is BSD-3-Clause",
    terms: "permissive",
    url: "https://github.com/networkx/networkx/blob/main/LICENSE.txt",
    host: "bundled",
    pages: ["/discrete-maths"],
  },
  {
    id: "transformers-js-docs",
    name: "Sample images, audio and video",
    credit:
      "The Transformers.js documentation's sample media (Xenova/transformers.js-docs), offered as one-click inputs on the audio and vision pages.",
    licence: "No licence stated by the collection",
    terms: "unstated",
    url: "https://huggingface.co/datasets/Xenova/transformers.js-docs",
    host: "huggingface",
    pages: ["audio and vision pages"],
  },
];
