// The CSV parse, off the main thread.
//
// Its own worker rather than a message on the fit worker, and the reason is the
// shared envelope: `ModelResponse`'s `ready` variant carries `{ model, backend }`
// and nothing else, so a parse that rode Machine A would have no way to hand the
// column list back — and the page needs the columns *before* it can offer a
// target to fit against. A one-shot request/response worker says that plainly
// instead of widening an envelope five other modalities depend on.
//
// A ten-megabyte file parsed on the main thread freezes the tab for a second,
// and the user's first interaction with the page is a stutter.

import { parseCsv, type ParseOptions } from "./csv";
import type { ParseResult } from "./types";

interface ParseRequest extends ParseOptions {
  id: number;
  text: string;
}

type ParseResponse =
  | { id: number; ok: true; result: ParseResult }
  | { id: number; ok: false; error: string };

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<ParseRequest>) => void) | null;
  postMessage: (message: ParseResponse, transfer?: Transferable[]) => void;
};

ctx.onmessage = (event) => {
  const { id, text, ...options } = event.data;
  try {
    const result = parseCsv(text, options);
    // Transferred, not copied: the page is about to become the only owner and
    // the arrays are the largest thing in the tab. The worker holds no
    // reference afterwards, which is the condition transfer actually requires.
    const transfer = result.dataset.columns.map((c) => c.values.buffer);
    ctx.postMessage({ id, ok: true, result }, transfer as Transferable[]);
  } catch (error) {
    ctx.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
