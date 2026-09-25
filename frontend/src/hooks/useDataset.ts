// The dataset the two Tabular pages share: a bundled sample, or the user's own
// file, parsed in a worker and held in memory.
//
// **Nothing here persists anything, and that is the page's whole subject.**
// `lib/mnistCache.ts` and `lib/proteinsCache.ts` both write their dataset to
// IndexedDB so a reload does not re-download it, and both are right to — they
// cache a public benchmark. This one would be caching someone's payroll. The
// difference between "the weights came to you" and "your file never left the
// device" is the only privacy claim in this app that covers data people
// actually mind about, and a convenience cache would quietly retract it.
//
// So: no IndexedDB, no `localStorage`, no upload, no run record. A reload loses
// the file, and re-dropping it costs a parse.

import { useCallback, useRef, useState } from "react";

import { parseCsvInWorker } from "@/tabular/client";
import { MAX_ROWS } from "@/tabular/limits";
import type { SampleDataset } from "@/tabular/samples";
import type { Dataset, ParseIssue } from "@/tabular/types";

export interface UseDatasetResult {
  dataset: Dataset | null;
  /** The bundled sample currently loaded, if the dataset came from one. */
  sample: SampleDataset | null;
  parsing: boolean;
  error: string | null;
  /** Rows the parser refused, with their line numbers. */
  issues: ParseIssue[];
  loadSample: (sample: SampleDataset) => Promise<void>;
  loadFile: (file: File) => Promise<void>;
  clear: () => void;
}

export function useDataset(
  parse: typeof parseCsvInWorker = parseCsvInWorker,
): UseDatasetResult {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [sample, setSample] = useState<SampleDataset | null>(null);
  const [parsing, setParsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [issues, setIssues] = useState<ParseIssue[]>([]);
  // Only the newest parse may write state. Dropping a second file while the
  // first is still parsing otherwise lands whichever finishes last.
  const generation = useRef(0);

  const run = useCallback(
    async (text: string, name: string, from: SampleDataset | null) => {
      const mine = ++generation.current;
      setParsing(true);
      setError(null);
      try {
        const result = await parse(text, { name, maxRows: MAX_ROWS });
        if (generation.current !== mine) return;
        setDataset(result.dataset);
        setIssues(result.issues);
        setSample(from);
      } catch (e) {
        if (generation.current !== mine) return;
        setError(e instanceof Error ? e.message : String(e));
        setDataset(null);
        setIssues([]);
        setSample(null);
      } finally {
        if (generation.current === mine) setParsing(false);
      }
    },
    [parse],
  );

  const loadSample = useCallback(
    async (s: SampleDataset) => {
      // The CSV itself is a dynamic import (see `samples.ts`), so choosing a
      // sample fetches a small chunk. That is a *file read*, not a model load:
      // nothing is fitted, and the FIT button is still the only thing that
      // spends.
      const text = await s.load();
      await run(text, s.label, s);
    },
    [run],
  );

  const loadFile = useCallback(
    async (file: File) => {
      const text = await file.text();
      await run(text, file.name, null);
    },
    [run],
  );

  const clear = useCallback(() => {
    generation.current++;
    setDataset(null);
    setSample(null);
    setIssues([]);
    setError(null);
  }, []);

  return { dataset, sample, parsing, error, issues, loadSample, loadFile, clear };
}
