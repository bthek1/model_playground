// Everything the app keeps on the visitor's device, and why (#62).
//
// The privacy notice renders this table. `storage.test.ts` scans the source
// for every `localStorage` key, Cache Storage bucket, IndexedDB database and
// persisted store name, and fails, naming the key, when one is missing here —
// an undisclosed key is how a notice quietly becomes untrue.
//
// None of it needs consent under the ePrivacy rules: each item either holds a
// choice the visitor made or stores something they asked for, so it is
// strictly necessary for the service requested. Analytics stores nothing at
// all (`persistence: "memory"`, #60), which is why there is no cookie banner.

export interface StoredItem {
  /** The exact key, bucket or database name. */
  key: string;
  where: "localStorage" | "Cache Storage" | "IndexedDB";
  /** What it holds and why, in a sentence. */
  purpose: string;
  /** Only written by a build with a backend to sign in to. */
  backendOnly?: boolean;
  /** Only written by a build that ships analytics. */
  analyticsOnly?: boolean;
}

export const STORED_ITEMS: readonly StoredItem[] = [
  {
    key: "theme",
    where: "localStorage",
    purpose: "Your light or dark theme choice.",
  },
  {
    key: "systemPanel",
    where: "localStorage",
    purpose: "Whether the system panel was left open.",
  },
  {
    key: "model-prefs",
    where: "localStorage",
    purpose:
      "The model you last selected on each page, so a refresh keeps the choice. It never reloads the model by itself.",
  },
  {
    key: "mp.analytics.optOut",
    where: "localStorage",
    purpose: "Set only if you turn analytics off, so the choice survives a reload.",
    analyticsOnly: true,
  },
  {
    key: "access_token",
    where: "localStorage",
    purpose: "Your sign-in session.",
    backendOnly: true,
  },
  {
    key: "refresh_token",
    where: "localStorage",
    purpose: "Renews your sign-in session.",
    backendOnly: true,
  },
  {
    key: "transformers-cache",
    where: "Cache Storage",
    purpose:
      "Model weights you chose to download, so the next load is free. The system panel shows its size and each page can clear a model's files.",
  },
  {
    key: "model-playground",
    where: "IndexedDB",
    purpose: "The MNIST digits, after the training page first downloads them.",
  },
  {
    key: "model-playground-proteins",
    where: "IndexedDB",
    purpose: "The PROTEINS dataset, after the graph-classification page first downloads it.",
  },
];
