# Graph Machine Learning in the Browser (WebGPU or CPU)

> The **Other → Graph Machine Learning** task of
> `components/layout/taskTaxonomy.ts`: what runs client-side on the user's GPU
> (WebGPU) or CPU (WebAssembly). No Python server in the inference path.

**Not built** — Graph Machine Learning renders the `/tasks/graph-machine-learning`
placeholder. This file is the research a plan gets written from; the procedure is
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).

Graph learning has no Hugging Face pipeline, no ONNX mirror, and no checkpoint
worth downloading. It also has, by some distance, the best claim on the raw
WebGPU path.

The reason is that a graph neural network is **one sparse matrix multiply
repeated a few times**. Message passing is `A_hat @ X @ W`: gather the
neighbours, average them, project. There is no attention stack, no tokenizer, no
processor config. The reference implementations build four GNNs from scratch for
exactly this reason, and building them again in WGSL is a comparable amount of
work rather than a port.

The other reason is that the datasets are tiny. Cora is 2708 nodes and 5429
edges, which is a few hundred kilobytes and fits in a `Float32Array`. Training a
2-layer GCN on it takes a second or two. So the page trains in front of the
user, and the whole graph is on screen while it does.

This is the one category where the browser genuinely beats a notebook, because a
graph is a picture and a notebook can only draw it once.

---

## 1. The core stack

```bash
# Nothing. No Transformers.js, no onnxruntime-web unless you export a trained model.
```

Three pieces, all of them already present or small enough to write:

| Piece | Implementation |
|---|---|
| The message-passing kernel | a new WGSL compute shader: gather, average, matmul |
| The dense projection | the existing [`webgpu/shaders/matmul.wgsl`](../../../frontend/src/webgpu/shaders/matmul.wgsl) |
| The layout, so the graph can be drawn | force-directed, in a Worker, or precomputed |

The pipeline is the one [`src/webgpu/`](../../../frontend/src/webgpu/) already
uses everywhere — `getGPUDevice()` then `createComputePipeline(wgsl)` then storage
buffers (`buffers.ts`) then `dispatchWorkgroups` then `readBackFloat32`, with
`runtime.ts` as the reference caller. Shaders are imported as strings via Vite's
`?raw`, and the heavy work runs in
[`webgpu/worker.ts`](../../../frontend/src/webgpu/worker.ts), never on the UI
thread.

**Cross-check every new kernel against a CPU reference.** A wrong aggregation
produces a plausible-looking accuracy curve, which is the same failure mode the
enhancement route's DSP had, and `tensorops.test.ts` is the pattern for catching
it.

---

## 2. Representing the graph on the GPU

Building the graph tensors is the part that transfers most directly from Python,
because CSR is CSR in any language.

```ts
// Compressed sparse row: the only sane way to hand a graph to a shader.
interface Graph {
  rowPtr: Uint32Array;    // length nNodes + 1
  colIdx: Uint32Array;    // length nEdges, the neighbours
  features: Float32Array; // nNodes * nFeatures, row-major
  labels: Uint8Array;     // nNodes
  trainMask: Uint8Array;  // the split. Cora uses 20 nodes per class
}
```

One workgroup per node, each thread walking its own neighbour slice:

```wgsl
// webgpu/shaders/gnn_aggregate.wgsl - symmetric-normalised mean aggregation
@group(0) @binding(0) var<storage, read>       rowPtr : array<u32>;
@group(0) @binding(1) var<storage, read>       colIdx : array<u32>;
@group(0) @binding(2) var<storage, read>       x      : array<f32>;
@group(0) @binding(3) var<storage, read_write> out    : array<f32>;
@group(0) @binding(4) var<uniform>             dims   : vec2<u32>;   // nNodes, nFeat

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let node = gid.x;
  if (node >= dims.x) { return; }
  let start = rowPtr[node];
  let end   = rowPtr[node + 1u];
  let deg   = f32(end - start) + 1.0;              // +1 for the self-loop
  for (var f = 0u; f < dims.y; f = f + 1u) {
    var acc = x[node * dims.y + f];                // the self-loop term
    for (var e = start; e < end; e = e + 1u) {
      acc = acc + x[colIdx[e] * dims.y + f];
    }
    out[node * dims.y + f] = acc / deg;
  }
}
```

Two notes on that shader, both of which are just `A_hat = D^-1 (A + I)` written
out:

- **The self-loop is not optional.** `A_hat = A + I`. Without it a node's own
  features are discarded at every layer, and the model can only see its
  neighbourhood.
- **The normalisation choice is visible in the output.** Mean aggregation is
  what is written above; GCN's symmetric `D^-1/2 A D^-1/2` needs the degree of
  both endpoints. Expose the choice as a control, because it changes the
  accuracy by a few points and that is a result worth showing.

---

## 3. What to build

### 3.1 Node classification on Cora, with the graph on screen

Building the tensors, then four GNNs from scratch.

The page: load Cora, lay it out, colour every node by its predicted class, and
train. As the epochs run, the colours settle into communities. Twenty labelled
nodes per class propagate outward through the graph, and watching that happen is
the clearest possible explanation of what semi-supervised node classification
means.

Ship all four architectures as the SELECT control: GCN, GraphSAGE, GAT and GIN.
They differ only in the aggregation step, which is one shader each, and comparing
them on the same layout with the same seed is a head-to-head made live.

Cora is small enough to bundle with the app, so there is nothing to download. The
LOAD slot still renders — it becomes
[`DeviceStatus`](../../../frontend/src/components/model/DeviceStatus.tsx), which
answers the question this page actually raises: is there a GPU, or nothing to
compute on. `/tensor` is the precedent, and `autoLoad` may be `true` because a
shader compile is fast and free.

### 3.2 Oversmoothing, the experiment that deserves a page of its own

The most valuable thing in this category.

Every deployed GNN is shallow, and the reason is that stacking layers drives
every node representation toward the same vector. A notebook demonstrates this
with a plot. A page demonstrates it with a **depth slider**: drag from 2 layers
to 8 and watch the node colours converge into an undifferentiated wash while the
accuracy falls.

That is a genuinely hard thing to convey in text, it is one control to build,
and it explains a design constraint that governs the entire field. If only one
page comes out of this category, this is the one.

### 3.3 Link prediction, the graph editing itself

Predict edges rather than labels: score node pairs by the dot product of their
embeddings, and draw the top predicted edges that are not in the graph as dashed
lines. Let the user click two nodes and see the score.

This is the page that makes the recommendation-system use case concrete, and it
needs no new kernel. The embeddings are already computed by 3.1.

### 3.4 Graph classification, one label per graph

Over `graphs-datasets/PROTEINS`.

A different pooling step, a different dataset, and a much smaller visual payoff:
the output is a single label per molecule rather than a colour per node. Worth
building for completeness after the first three, not before.

The dataset is small enough to fetch from the Hub at runtime as JSON, so there
is a real LOAD state here even though there are no weights.

---

## 4. Feasibility summary

| Piece | In-browser? | How | Best backend |
|---|---|---|---|
| **Building the graph tensors** | Yes | CSR arrays, bundled with the app | main thread |
| **Four GNNs from scratch** | Yes, excellent | one WGSL aggregation shader per architecture | WebGPU |
| **Link prediction** | Yes | dot products over the node embeddings | WebGPU |
| **Oversmoothing** | Yes, and it is the best page here | a depth slider over the same model | WebGPU |
| **Graph classification** | Yes | pooling plus a small dataset fetch | WebGPU |
| Large-graph training (millions of nodes) | No | sampling and partitioning are a different system | server |

There is no "if not" column, because unusually there is no server fallback to
recommend. Everything at this scale runs in a tab.

---

## 5. Memory and performance notes

- **CSR, never an adjacency matrix.** Cora as a dense `2708 x 2708` matrix is
  29 MB of mostly zeros; as CSR it is around 40 KB. On any graph worth drawing,
  the dense form is the wrong shape before it is a performance problem.
- **Upload the graph once.** `rowPtr` and `colIdx` never change during training,
  so they are written to storage buffers at load and left there. Only the
  feature and weight buffers are touched per epoch.
- **The layout is the expensive part, not the model.** A force-directed layout
  over a few thousand nodes costs more than the GNN does. Compute it once in a
  Worker, cache the coordinates, and never recompute on a hyperparameter change.
- **Render with a single canvas draw**, not a DOM node per graph node. Two
  thousand SVG circles will not animate; a canvas with two thousand arcs will.
- **Read back once per epoch at most.** `readBackFloat32` is a GPU-to-CPU sync
  and it stalls the pipeline. Train several epochs, then read the embeddings for
  rendering.
- **Nothing to dispose.** No downloaded weights and no pretrained session. The
  GPU device is memoised and shared, exactly as the existing `/tensor` and
  `/training` routes already use it.

---

## 6. Reference

- **[`src/webgpu/`](../../../frontend/src/webgpu/)**: `getGPUDevice`,
  `createComputePipeline`, `readBackFloat32`, the existing matmul and elementwise
  shaders, and `webgpu/worker.ts`. This page is a direct extension of that path,
  not of the Transformers.js one — and the two never mix in one directory.
- **`detectWebGPU()`** ([`webgpu/capabilities.ts`](../../../frontend/src/webgpu/capabilities.ts))
  matters more here than anywhere else, because there is no WASM fallback for a
  hand-written shader. It never throws — it returns `unsupported` / `no-adapter` /
  `no-device` / `ready` — and `ready` means a `GPUDevice` was actually acquired.
  Note the common false negative: `navigator.gpu` needs a **secure context**, so a
  plain-HTTP LAN origin hides WebGPU entirely, and Firefox needs
  `dom.webgpu.enabled`. See
  [`../../explanations/webgpu-inference.md`](../../explanations/webgpu-inference.md).
- **Datasets**: Cora (bundle it, it is tiny) and `graphs-datasets/PROTEINS`
  (fetch at runtime).
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md),
  and [`../../standards/model-visualization.md`](../../standards/model-visualization.md)
  for how the graph itself gets drawn — canvas, theme tokens, diverging colour.
- The method recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo.
  Nothing on this page needs a checkpoint at all.
