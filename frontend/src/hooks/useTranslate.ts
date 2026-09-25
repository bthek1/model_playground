// Translation. A thin wrapper over the generic text worker — it picks the pair
// from the catalogue, and that is nearly all there is to it, because a Marian
// checkpoint *is* the language pair.
//
// The thing worth saying out loud: **there is no language argument.** `tr(text)`
// takes nothing but the text. A hook that accepted `{ from, to }` and quietly
// dropped them would look like it worked — the model would keep translating in
// the direction it was built for — which is exactly why the direction lives in
// SELECT as a model choice and this hook takes no language at all.

import { useCallback, useMemo, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import {
  DEFAULT_TRANSLATION_MODEL,
  TRANSLATION_MODELS,
  type TranslationModel,
} from "@/text/catalogue";

/** What one Marian call returns, per input. */
interface RawTranslation {
  translation_text?: string;
}

export interface UseTranslateResult extends Omit<UseTextPipelineResult, "run"> {
  /** The selected pair. The direction comes from here, never from an argument. */
  meta: TranslationModel;
  /** The latest translation. */
  result: string | null;
  /**
   * Translate one text. No language arguments — see the header.
   *
   * `max_new_tokens` is a **run** parameter, so asking for a longer output is a
   * different question about the same weights and must never imply another
   * download.
   */
  run: (text: string, opts?: { maxNewTokens?: number }) => Promise<string>;
}

/**
 * Generated-token ceiling.
 *
 * Marian's own config has no `max_length` worth relying on, and a seq2seq that
 * never emits EOS decodes until something stops it — on a degenerate input that
 * is a minute of repeated text rather than an error. 512 is well past any
 * paragraph a user will paste and still bounded.
 */
export const MAX_NEW_TOKENS = 512;

export function useTranslate(
  model: string = DEFAULT_TRANSLATION_MODEL,
  autoLoad = false,
): UseTranslateResult {
  const meta = useMemo(
    () =>
      TRANSLATION_MODELS.find((m) => m.id === model) ?? TRANSLATION_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<string | null>(null);

  const run = useCallback(
    async (text: string, opts?: { maxNewTokens?: number }): Promise<string> => {
      const raw = (await post(text, [
        { max_new_tokens: opts?.maxNewTokens ?? MAX_NEW_TOKENS },
      ])) as RawTranslation[] | RawTranslation | undefined;
      const first = Array.isArray(raw) ? raw[0] : raw;
      const out = String(first?.translation_text ?? "");
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, meta, result, run };
}
