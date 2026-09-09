// Making a pipeline's output survive `postMessage`.
//
// The same trap as `RawImage` on the way *in*, on the way back out and one level
// worse. A Transformers.js `Tensor` exposes `data` and `dims` as **getters on its
// prototype**, over an internal ONNX Runtime tensor; the structured-clone
// algorithm copies own properties and refuses the rest, so posting one fails
// outright with:
//
//     Failed to execute 'postMessage' on 'DedicatedWorkerGlobalScope':
//     #<_Tensor> could not be cloned.
//
// Depth estimation is the task that hits it — `{ predicted_depth: Tensor, … }` —
// and it is invisible to the unit suite, which mocks the worker away entirely,
// and to a mocked E2E run, which never loads a model. It took a real load to
// surface, which is what the `@slow` specs are for.
//
// The conversion is duck-typed rather than `instanceof`, deliberately: that keeps
// this module free of `@huggingface/transformers`, so `engine.ts` stays cheap to
// unit-test without the runtime.

/** A tensor flattened to something the structured-clone algorithm accepts. */
export interface PlainTensor {
  data: ArrayLike<number>;
  dims: number[];
  type?: string;
}

/** A single- or multi-channel image, flattened the same way. */
export interface PlainImage {
  data: ArrayLike<number>;
  width: number;
  height: number;
  channels: number;
}

function isTypedArray(value: unknown): boolean {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function has(value: object, key: string): boolean {
  // `in`, not `hasOwnProperty`: `data` and `dims` are prototype getters on a
  // Tensor, which is the whole reason this file exists.
  return key in value;
}

/**
 * Deep-convert a pipeline result into plain, cloneable data.
 *
 * Anything already cloneable is passed through untouched. Tensors and images are
 * flattened to their numbers plus their shape; `data` is **copied** with
 * `slice()` so the posted buffer does not alias a view the runtime may reuse for
 * the next inference.
 */
export function toCloneable(value: unknown, depth = 0): unknown {
  if (value === null || typeof value !== "object") return value;
  if (isTypedArray(value) || value instanceof ArrayBuffer) return value;
  // A malformed or self-referential result must not hang the worker.
  if (depth > 8) return null;

  if (Array.isArray(value)) {
    return value.map((item) => toCloneable(item, depth + 1));
  }

  const obj = value as Record<string, unknown>;

  // Tensor first: it also carries `data`, so an image check would claim it.
  if (has(obj, "dims") && has(obj, "data")) {
    const dims = obj.dims;
    return {
      data: copyNumbers(obj.data),
      dims: Array.isArray(dims) ? [...(dims as number[])] : [],
      ...(typeof obj.type === "string" ? { type: obj.type } : {}),
    } satisfies PlainTensor;
  }

  if (
    has(obj, "data") &&
    has(obj, "width") &&
    has(obj, "height") &&
    has(obj, "channels")
  ) {
    return {
      data: copyNumbers(obj.data),
      width: Number(obj.width),
      height: Number(obj.height),
      channels: Number(obj.channels),
    } satisfies PlainImage;
  }

  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(obj)) {
    out[key] = toCloneable(item, depth + 1);
  }
  return out;
}

function copyNumbers(data: unknown): ArrayLike<number> {
  if (isTypedArray(data)) {
    return (data as unknown as { slice: () => ArrayLike<number> }).slice();
  }
  if (Array.isArray(data)) return [...(data as number[])];
  return [];
}
