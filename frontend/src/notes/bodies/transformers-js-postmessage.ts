import type { NoteBlock } from "../types";

export const body: readonly NoteBlock[] = [
  {
    kind: "p",
    text: "Running Transformers.js in a Web Worker means moving inputs and results across `postMessage`, which uses structured cloning. Two of the library's types do not survive it. A `Tensor` fails outright with `#<_Tensor> could not be cloned`. A `RawImage` clones, but arrives as a plain object with no methods, and the pipeline rejects it.",
  },
  { kind: "h2", text: "Why" },
  {
    kind: "p",
    text: "Structured cloning copies own data properties and drops prototypes. A `Tensor`'s `data` and `dims` are getters on the prototype, reading from an internal ONNX Runtime tensor object, and structured cloning refuses the whole object. A `RawImage` keeps its pixels in own properties, so those arrive, but the class is gone.",
  },
  { kind: "h2", text: "The fix: send plain data, rebuild on the other side" },
  {
    kind: "code",
    text: `import { RawImage } from "@huggingface/transformers";

// Page: an image becomes typed-array pixels plus its shape.
const payload = {
  data: image.data.slice(), // a copy: the page may still be showing it
  width: image.width,
  height: image.height,
  channels: image.channels,
};
worker.postMessage(payload, [payload.data.buffer]);

// Worker: rebuild the class before calling the pipeline.
const input = new RawImage(payload.data, payload.width, payload.height, payload.channels);

// Worker: flatten any Tensor in the result before posting it back.
const flat = { data: tensor.data, dims: tensor.dims, type: tensor.type };
self.postMessage(flat, [flat.data.buffer]);`,
  },
  {
    kind: "p",
    text: "Results can nest tensors, as depth estimation does with its `predicted_depth`, so walk the result and flatten every `Tensor` before posting, rather than handling one known field.",
  },
  { kind: "h2", text: "Copy or transfer?" },
  {
    kind: "p",
    text: "Transferring a buffer is free, but it detaches the buffer on the sending side. If the page is still showing the image it just sent, the preview goes blank. If the same recording is sent twice, the second send has nothing to send. Copy by default, and transfer only data you will not touch again, such as a webcam frame already drawn.",
  },
  { kind: "h2", text: "Why your tests did not catch it" },
  {
    kind: "p",
    text: "A unit test usually replaces the worker with a function call that passes objects by reference, so nothing is cloned. A mocked end-to-end run loads no weights, so no real `Tensor` is produced. On this site the depth page was the first to return a tensor, and only a real model load showed the error. If you test a worker with a harness, make the harness call `structuredClone(message, { transfer })` the way `postMessage` does.",
  },
  {
    kind: "p",
    text: "Pages on this site that send images to a worker this way: [depth estimation](/depth), [image classification](/image-classification) and [image embeddings](/image-features). The first and last also get tensors back.",
  },
];
