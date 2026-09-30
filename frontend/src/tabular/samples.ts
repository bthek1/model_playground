// The bundled sample datasets.
//
// A page whose only input is "give me your payroll file" is a page nobody tries.
// These exist so the ladder can be run, compared and disbelieved before the user
// trusts the tab with anything of their own — and they are committed rather than
// fetched, because a route whose whole claim is "nothing leaves your device"
// should not open by making a network request.
//
// Every entry states its licence. #24 is the precedent for that being load-
// bearing rather than decorative: a licence is the one thing that can invalidate
// a finished route, and it is cheaper to read the card first.

// **Loaded lazily, and that is load-bearing.** Three CSVs are ~97 KB of text
// that no minifier can compress, and a static `?raw` import puts every byte of
// all three into the entry chunk — `npm run check:bundle` fails on exactly that,
// which is what it is for. A dynamic import splits each into its own chunk,
// fetched when the sample is chosen and never on a cold first paint. It works
// here only because nothing imports these files statically: one static importer
// anywhere and the dynamic import buys a microtask and nothing else
// (`[INEFFECTIVE_DYNAMIC_IMPORT]`, a build *warning*, which survives a green
// build indefinitely).

export interface SampleDataset {
  id: string;
  label: string;
  /** One sentence: what it is and what it is here to show. */
  blurb: string;
  /** Fetches the CSV text. Lazy — see the note at the top of this file. */
  load: () => Promise<string>;
  licence: string;
  source: string;
  /** Column name to preselect as the classification target. */
  target: string;
  /** Column name to preselect as the regression target, where one makes sense. */
  regressionTarget?: string;
  /** True for data that was generated rather than observed. Rendered as such. */
  synthetic?: boolean;
}

export const SAMPLES: SampleDataset[] = [
  {
    id: "credit-risk",
    label: "Credit risk (synthetic)",
    blurb:
      "1 800 loans whose risk is an *interaction* — being leveraged is dangerous on a low income and harmless on a high one. No straight line can express that, so the ladder disagrees with itself here, which is the only way this page says anything about four models rather than one.",
    load: () => import("./data/credit-risk-synthetic.csv?raw").then((m) => m.default),
    licence: "Generated in this repo — see scripts/make-tabular-samples.mjs",
    source: "Synthetic, fixed seed",
    target: "defaulted",
    regressionTarget: "debt_ratio",
    synthetic: true,
  },
  {
    id: "penguins",
    label: "Palmer penguins",
    blurb:
      "344 birds measured in the Palmer Archipelago. Real missing values, a categorical island column and a three-species target — and separable enough that the floor of the ladder wins, which is worth seeing too.",
    load: () => import("./data/penguins.csv?raw").then((m) => m.default),
    licence: "CC0 1.0 (public domain dedication)",
    source: "Horst, Hill & Gorman (2020), palmerpenguins",
    target: "species",
    regressionTarget: "body_mass_g",
  },
  {
    id: "wine",
    label: "Wine recognition",
    blurb:
      "178 wines, thirteen chemistry columns, three cultivars. The classic small benchmark: every family on the ladder does well, so it is the control rather than the demonstration.",
    load: () => import("./data/wine.csv?raw").then((m) => m.default),
    licence: "CC BY 4.0",
    source: "Forina et al. (1991), UCI Machine Learning Repository",
    target: "cultivar",
    regressionTarget: "proline",
  },
];

export function sampleById(id: string): SampleDataset | undefined {
  return SAMPLES.find((s) => s.id === id);
}
