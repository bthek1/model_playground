import { AutoTokenizer, AutoModelForQuestionAnswering } from "@huggingface/transformers";

const ID = "Xenova/distilbert-base-cased-distilled-squad";
const tok = await AutoTokenizer.from_pretrained(ID);
const model = await AutoModelForQuestionAnswering.from_pretrained(ID, { dtype: "q8" });

const SEP = tok.sep_token_id ?? 102;

function softmax(a) {
  const m = Math.max(...a.filter(Number.isFinite));
  const e = a.map((x) => (Number.isFinite(x) ? Math.exp(x - m) : 0));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map((x) => x / s);
}

function align(text, toks) {
  const out = [];
  let pos = 0;
  for (const raw of toks) {
    const piece = raw.startsWith("##") ? raw.slice(2) : raw;
    while (pos < text.length && /\s/.test(text[pos])) pos += 1;
    if (text.slice(pos, pos + piece.length) !== piece) return null;
    out.push([pos, pos + piece.length]);
    pos += piece.length;
  }
  return out;
}

async function ask(question, context) {
  const inputs = tok(question, { text_pair: context, padding: true, truncation: true });
  const { start_logits, end_logits } = await model(inputs);
  const ids = inputs.input_ids.tolist()[0];
  const sepIndex = ids.findIndex((x) => x == SEP);
  const lastSep = ids.length - 1 - [...ids].reverse().findIndex((x) => x == SEP);

  const start = start_logits[0].tolist();
  const end = end_logits[0].tolist();
  for (let i = 0; i < start.length; ++i) {
    if (i <= sepIndex || i >= lastSep) { start[i] = -Infinity; end[i] = -Infinity; }
  }
  const sp = softmax(start), ep = softmax(end);
  let best = null;
  const MAX_LEN = 30;
  for (let i = sepIndex + 1; i < lastSep; ++i) {
    for (let j = i; j < Math.min(i + MAX_LEN, lastSep); ++j) {
      const sc = sp[i] * ep[j];
      if (!best || sc > best.score) best = { i, j, score: sc };
    }
  }
  const ctxTokens = ids.slice(sepIndex + 1, lastSep).map((x) => tok.decode([x], { skip_special_tokens: false }));
  const offs = align(context, ctxTokens);
  const answerTokens = ids.slice(best.i, best.j + 1);
  const decoded = tok.decode(answerTokens, { skip_special_tokens: true });
  if (!offs) return { decoded, score: best.score, span: null };
  const s = offs[best.i - sepIndex - 1][0];
  const e = offs[best.j - sepIndex - 1][1];
  return { decoded, score: best.score, span: [s, e], sliced: context.slice(s, e) };
}

const EIFFEL = "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris. It stood as the world's tallest man-made structure for 41 years, until the Chrysler Building in New York was finished in 1930.";
const WEBGPU = "WebGPU is a web standard that exposes modern GPU capabilities to the browser. It was first shipped in Chrome 113 in May 2023, and is developed by the W3C GPU for the Web Community Group. Unlike WebGL, it supports general-purpose compute shaders.";

for (const [q, c] of [
  ["Who built the Eiffel Tower?", EIFFEL],
  ["How long was it the tallest structure?", EIFFEL],
  ["When did WebGPU first ship?", WEBGPU],
  ["What does it support that WebGL does not?", WEBGPU],
  // The unanswerable one.
  ["How much does a WebGPU licence cost?", WEBGPU],
  ["Who won the 1998 World Cup?", EIFFEL],
]) {
  const r = await ask(q, c);
  console.log(JSON.stringify(q), "->", JSON.stringify(r));
}
