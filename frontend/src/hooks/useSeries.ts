// The series the forecasting page holds: a bundled sample, a pasted column, or
// a dropped file.
//
// **There is no worker here, and that is a decision rather than an omission.**
// Naive, seasonal naive and drift are arithmetic over one array and a rolling
// backtest is a loop over slices — the whole page is O(points × windows) on a
// series of a few thousand points. Wrapping pure arithmetic in a worker to look
// consistent with the other modalities would add a protocol, a mock and a
// second state machine for no benefit, and would give the page an asynchronous
// boundary it has nothing to put across.
//
// So the parse is synchronous, the text box *is* the state, and every control
// downstream re-derives with `useMemo`.

import { useCallback, useMemo, useState } from "react";

import type { ForecastSample } from "@/forecast/samples";
import { parseSeries, type Series } from "@/forecast/series";

export interface UseSeriesResult {
  /** The raw text in the box — the source of truth. */
  text: string;
  setText: (value: string) => void;
  /** Parsed, or null while the text does not parse. */
  series: Series | null;
  error: string | null;
  /** The bundled sample currently loaded, if the text came from one. */
  sample: ForecastSample | null;
  loadSample: (sample: ForecastSample) => Promise<void>;
  loadFile: (file: File) => Promise<void>;
}

export function useSeries(): UseSeriesResult {
  const [text, setTextRaw] = useState("");
  const [name, setName] = useState("series");
  const [sample, setSample] = useState<ForecastSample | null>(null);

  const setText = useCallback((value: string) => {
    setTextRaw(value);
    // Typing detaches the text from whatever sample it came from — otherwise
    // the panel keeps crediting a licence and a source to data the user has
    // since edited.
    setSample(null);
    setName("pasted series");
  }, []);

  const loadSample = useCallback(async (s: ForecastSample) => {
    const csv = await s.load();
    setTextRaw(csv);
    setName(s.label);
    setSample(s);
  }, []);

  const loadFile = useCallback(async (file: File) => {
    const content = await file.text();
    setTextRaw(content);
    setName(file.name);
    setSample(null);
  }, []);

  // Parsing on every keystroke is fine at this size and is what makes the gap
  // and irregularity report live: the user sees "4 gaps" while they are still
  // looking at the paste that caused them.
  const parsed = useMemo(() => {
    if (text.trim().length === 0) return { series: null, error: null };
    try {
      return { series: parseSeries(text, name), error: null };
    } catch (e) {
      return { series: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [text, name]);

  return {
    text,
    setText,
    series: parsed.series,
    error: parsed.error,
    sample,
    loadSample,
    loadFile,
  };
}
