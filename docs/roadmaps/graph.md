# Graph Machine Learning in the Browser (WebGPU or CPU)

> The **Other → Graph Machine Learning** task of
> [`taskTaxonomy.ts`](../../frontend/src/components/layout/taskTaxonomy.ts): what runs
> client-side on the user's GPU (WebGPU) or CPU. No Python server in the inference path,
> and — unusually — no checkpoint either.
>
> This file began as issue [#3](https://github.com/bthek1/model_playground/issues/3) and
> moved here when its first route shipped, the same way
> [`audio.md`](audio.md) and [`vision.md`](vision.md) did: a roadmap that documents
> *shipped* code has to be reviewable in the same pull request as the code it describes,
> which an issue body cannot be. The build-out is recorded in the closed plan issue
> [#26](https://github.com/bthek1/model_playground/issues/26).

| Roadmap section | Route | Status |
|---|---|---|
| Node classification on Cora (§3.1) | [`/graph`](../../frontend/src/routes/graph.tsx) | **Shipped** — GCN, GraphSAGE, GIN, GAT |
| Oversmoothing (§3.2) | [`/graph`](../../frontend/src/routes/graph.tsx) | **Shipped** — the depth slider, and a number behind it |
| Link prediction (§3.3) | — | Not built; needs no new kernel |
| Graph classification (§3.4) | — | Not built; needs pooling and a runtime dataset fetch |

**This category is the strongest case in the repo for the raw-WebGPU path**, and the
reason is that a graph neural network is one sparse gather repeated a few times. There is
no Hugging Face pipeline, no ONNX mirror and no checkpoint worth downloading — so the
model is written as WGSL and trained in front of the user, which is also the only
category where that is the *better* option rather than the only one.

The datasets are small enough to make it work. Cora is 2708 nodes and 5278 citations, it
ships inside the app, and a 2-layer GCN trains on it in a couple of seconds. So the whole
graph is on screen while it learns — and that is the thing a notebook cannot do, because
a notebook can only draw the graph once.

---

## 1. The core stack

Nothing was installed. No Transformers.js, no `onnxruntime-web`; this route shares the
existing raw-WebGPU pipeline and adds one shader.

| Piece | Where |
|---|---|
| The dataset | [`lib/cora.ts`](../../frontend/src/lib/cora.ts) + `lib/data/cora.bin`, built by [`scripts/prepare-cora.mjs`](../../frontend/scripts/prepare-cora.mjs) |
| The layout | [`lib/graphLayout.ts`](../../frontend/src/lib/graphLayout.ts) |
| The message-passing kernel | [`webgpu/shaders/gnn_aggregate.wgsl`](../../frontend/src/webgpu/shaders/gnn_aggregate.wgsl) + [`webgpu/gnnRuntime.ts`](../../frontend/src/webgpu/gnnRuntime.ts) |
| The dense projection | the existing [`webgpu/shaders/matmul.wgsl`](../../frontend/src/webgpu/shaders/matmul.wgsl), unchanged |
| The model and its backward pass | [`webgpu/gnn.ts`](../../frontend/src/webgpu/gnn.ts), and [`webgpu/gat.ts`](../../frontend/src/webgpu/gat.ts) for attention |
| The worker-side session | [`webgpu/graphSession.ts`](../../frontend/src/webgpu/graphSession.ts) |
| The page | [`routes/graph.tsx`](../../frontend/src/routes/graph.tsx), [`hooks/useGraphTraining.ts`](../../frontend/src/hooks/useGraphTraining.ts), [`components/graph/GraphCanvas.tsx`](../../frontend/src/components/graph/GraphCanvas.tsx) |

The pipeline is the one `src/webgpu/` already uses everywhere — `getGPUDevice()` →
`createComputePipeline(wgsl)` → storage buffers → `dispatchWorkgroups` →
`readBackFloat32` — with `runtime.ts` as the reference caller. Heavy work runs in
[`webgpu/worker.ts`](../../frontend/src/webgpu/worker.ts), never on the UI thread.

---

## 2. Message passing is one scaled gather

Everything except GAT reduces to:

```
out[i] = α_i · Σ_{j ∈ N(i) ∪ {i}} β_j · x[j]
```

a **scaled gather** over the graph's CSR arrays, and each architecture is a choice of the
two scale vectors:

| Architecture | α_i | β_j | Aggregation |
|---|---|---|---|
| **GCN** | `1/√(deg_i+1)` | `1/√(deg_j+1)` | symmetric, `D^-1/2 (A+I) D^-1/2` |
| **GraphSAGE** | `1/(deg_i+1)` | `1` | mean, `D^-1 (A+I)` |
| **GIN** | `1` | `1` | sum, `(A+I)`, ε = 0 |

Four things about that formulation are load-bearing, and three of them fail silently.

- **The self-loop is not stored — it is added in the kernel.** `A_hat = A + I`. Without
  it a node's own features are discarded at every layer and the model can only see its
  neighbourhood. `cora.bin` therefore holds *no* self-loops, so the term cannot be
  counted twice, and `cora.test.ts` asserts that over the real 2708-node graph.
- **The transpose the backward pass needs is the same kernel with α and β swapped**,
  because `A + I` is symmetric: `(α_i β_j)ᵀ = α_j β_i`. That is why one shader serves
  forward and backward for all three architectures — and why the same test file asserts
  the graph really is symmetric. If it were not, the gradient would be quietly wrong.
- **Project before you gather.** `Â(XW)`, never `(ÂX)W`: aggregating a 2708×16 matrix
  instead of a 2708×1433 one is the difference between a responsive page and a stall.
- **The normalisation choice is visible in the result**, which is why it is a control
  rather than a constant. On our split, at 2 layers: GraphSAGE 0.80, GCN 0.78, GIN 0.74,
  GAT 0.69.

### GAT is not one of them, and the roadmap was wrong about that

The original research for this category said the four architectures "differ only in the
aggregation step, which is one shader each". That holds for GCN, GraphSAGE and GIN. It
does not hold for GAT, and finding that out is one of the two results of this work.

GAT's coefficient on an edge is *computed from the features being propagated*:

```
e_ij = LeakyReLU(a_srcᵀ s_i + a_dstᵀ s_j)      α_ij = softmax over j ∈ N(i) ∪ {i}
```

so it is per **edge** rather than a product of two per-node scales, the layer owns two
learnable vectors, and the gradient flows back into the same features twice — once
through the values and once through the softmax that weighted them. None of that is a
scale vector.

The seam that keeps the trainer's loops identical across all four is
[`Propagator`](../../frontend/src/webgpu/gnn.ts): a layer's message-passing step, with
`forward`, `backward` and any parameters it owns. A scaled gather implements it in three
lines; [`gat.ts`](../../frontend/src/webgpu/gat.ts) implements it in a hundred.

**GAT's gather deliberately does not use the shader**, and the page says so. It is
O(|E|·d) — 10 556 edges by 16 features, about 0.2 ms — while the projection beside it is
2708×1433×16, and *that* still runs on the GPU through the shared matmul. Moving a fifth
of a millisecond would mean carrying a per-edge softmax and its backward pass into WGSL.

One thing that usually makes attention awkward on a CSR graph did not arise: the backward
pass needs no reverse-edge index, because every term that would need one is an
accumulation into a dense per-node buffer, and the loop already visits each directed edge
exactly once.

---

## 3. What is built, and what is not

### 3.1 Node classification on Cora — **shipped**

2708 machine-learning papers, 5278 citations, a 1433-word binary bag-of-words each, seven
topics, and **20 labelled papers per topic** — 140 out of 2708. Load the graph, lay it
out, colour every node by its predicted class, and train: the colours settle into
communities as the labels propagate outward, which is the clearest available explanation
of what semi-supervised node classification means.

All four architectures are in SELECT. Depth, 2 to 8, is in RUN.

**Cora is bundled, sparse-encoded.** Dense float32 features are 2708 × 1433 × 4 = 15.5 MB
and unbundleable; the matrix is 1.27 % dense and every stored value is 1, so CSR with u16
column indices and no values array is **161 KB**. `scripts/prepare-cora.mjs` builds it
from the LINQS release and prints the header it produced, so the committed binary is
reproducible rather than a mystery blob. `decodeCora` refuses a bad magic number, an
unknown version, a truncated file or trailing bytes — a graph silently decoded at the
wrong offset still trains and still draws an accuracy curve.

The split is **ours, not Planetoid's**: the published split is a fixed index list over
Planetoid's node ordering, and this file's ordering comes from `cora.content`. Same recipe
(20 per class / 500 / 1000), same ballpark, not the same numbers as a published table.

Two implementation notes worth keeping:

- **Input dropout is worth several points, and it is affordable only because the features
  are sparse.** The first layer's weight gradient is `Xᵀ · dS`, so a masked input needs a
  masked transpose, and re-masking 3.9 M elements per epoch would cost more than the
  model. `prepareInput` records where the 49 216 nonzeros land in *both* layouts, so
  masking is 50 k writes. Without it a 2-layer GCN scores 0.754; with it, 0.776.
- **The layout is the expensive part, not the model** — about 900 ms against a couple of
  seconds of training, and it is computed once in the worker and never recomputed.
  Changing the architecture or the depth re-trains; it must never re-lay-out, or the user
  loses the mental map they were reading the result against.

### 3.2 Oversmoothing — **shipped**, and it needed a number

The most valuable thing in this category, and the reason the page exists. Every deployed
GNN is shallow, because stacking layers drives every node representation toward the same
vector. A notebook shows that with a plot; the page shows it with a **depth slider**.

Measured on the real graph, GCN, 200 epochs:

| Layers | Test accuracy | Neighbour similarity |
|---|---|---|
| 2 | 0.776 | 0.947 |
| 4 | 0.687 | 0.958 |
| 6 | 0.496 | 0.973 |
| 8 | 0.518 | 0.978 |

**The metric took two attempts, and the first one was wrong.** Mean pairwise cosine over
*all* node pairs — the obvious reading of "every representation converges" — does not move
monotonically with depth on real Cora (0.720, 0.643, 0.663, 0.579, 0.677). It is dominated
by how well the model separates classes, not by how much the graph has smoothed. What does
move monotonically is the similarity between **adjacent** nodes, which is the quantity
repeated message passing actually acts on, and which is the Dirichlet energy of the
row-normalised representations up to a constant — the literature's measure. Rows are
normalised first so a representation that merely *shrinks* with depth is not mistaken for
one that has smoothed.

**Depth is a RUN control and re-runs on purpose.** The page-pattern test is "does this
change what the model is asked?" — depth does, so the slider belongs beside the input and
re-training is correct. The colour switch beside the graph re-reads the result in hand and
never re-runs. The page says both.

**GIN's collapse is not oversmoothing, and the page refuses to let it look like one.**
Unnormalised sum multiplies activations by roughly the mean degree at every layer, so past
four or five layers it overflows and ReLU zeroes everything: at 8 layers test accuracy is
0.064 and *every* representation is dead. Smoothness is undefined there — there are no
directions left to compare — so `GnnMetrics.deadFraction` is reported separately and the
page names the failure when more than half the representations are gone.

### 3.3 Link prediction — not built

Score node pairs by the dot product of their embeddings and draw the top predicted edges
that are not in the graph as dashed lines. Needs **no new kernel** — the embeddings are
already computed by 3.1 — and no new dataset. This is the cheapest remaining page here.

### 3.4 Graph classification — not built

Over `graphs-datasets/PROTEINS`: a different pooling step, a different dataset, and a much
smaller visual payoff (one label per molecule rather than a colour per node). The dataset
is small enough to fetch from the Hub at runtime as JSON, so this is the one page in the
category with a real LOAD state even though there are still no weights.

---

## 4. Feasibility summary

| Piece | In-browser? | How | Best backend |
|---|---|---|---|
| **Building the graph tensors** | Yes | CSR, bundled with the app | main thread / worker |
| **GCN, GraphSAGE, GIN** | **Shipped** | one WGSL scaled-gather shader for all three | WebGPU |
| **GAT** | **Shipped** | attention in TS, projections on the GPU | WebGPU + CPU |
| **Oversmoothing** | **Shipped**, and the best page here | a depth slider over the same model | WebGPU |
| **Link prediction** | Yes | dot products over the node embeddings | WebGPU |
| **Graph classification** | Yes | pooling plus a small dataset fetch | WebGPU |
| Large-graph training (millions of nodes) | No | sampling and partitioning are a different system | server |

There is no "if not" column, because unusually there is no server fallback to recommend.
Everything at this scale runs in a tab.

**The CPU path here is not a fallback in the usual sense.** A hand-written shader has no
ONNX Runtime to fall back to — but the CPU aggregation *is* the reference implementation
the WGSL kernel is checked against, so a machine without WebGPU runs the identical
arithmetic, more slowly. The page says which one a run will use before it starts.

---

## 5. Memory and performance notes

- **CSR, never an adjacency matrix.** Cora as a dense 2708 × 2708 f32 matrix is 29 MB of
  mostly zeros, against about 40 KB as CSR. On any graph worth drawing the dense form is
  the wrong shape long before it is slow.
- **Upload the graph once.** `rowPtr`, `colIdx` and the scale vectors are written to
  storage buffers when the aggregator is created and left there for the whole run; only
  features and weights move per epoch. Output buffers are cached by size, so a run
  allocates two of them rather than two per layer per epoch.
- **The features never cross `postMessage`.** They are 15.5 MB dense, the page has no use
  for them, and the worker fetches and decodes the dataset itself. What the page receives
  is what it draws: coordinates, labels, the train mask and the CSR arrays — about 80 KB.
- **Read back once per epoch at most.** `readBackFloat32` is a GPU→CPU sync that stalls
  the pipeline. The canvas wants predictions per epoch, not per step, and metric updates
  are batched to animation frames on the main thread.
- **Render with a single canvas draw**, not a DOM node per graph node: 2708 SVG circles
  will not animate. The edges are painted once into an offscreen layer and blitted,
  because they never change — only the colours do.
- **Draw at a fixed square and pan/zoom it**, rather than redrawing to fit the column.
  2708 dots in a ~480 px box is a picture of the whole graph and nothing else, and the
  claim this page makes is that you can go and look at a community. The canvas is painted
  at 720 px inside [`viz/PanZoom.tsx`](../../frontend/src/components/viz/PanZoom.tsx)
  (shared with `/training`), which also supersamples it at rest. The *viewport* carries
  the frame: a canvas with its own border draws a second frame that pans away.
- **Nothing to dispose but buffers.** No downloaded weights and no pretrained session. The
  GPU device is memoised and shared, exactly as `/tensor` and `/training` already use it,
  and every buffer is freed with `releaseBuffer` so the system panel's ledger stays honest.

---

## 6. Testing, and what each level actually catches

A wrong aggregation produces a falling loss and a plausible accuracy curve. That is this
category's failure mode, and it is the same one `audio/enhance/`'s DSP had, so the tests
are arranged around it.

| Level | What it catches |
|---|---|
| [`lib/cora.test.ts`](../../frontend/src/lib/cora.test.ts) | a mis-parsed dataset: asymmetric edges, a stored self-loop, a bad split, features that don't normalise — asserted over the **real** committed binary, not a fixture |
| [`lib/graphLayout.test.ts`](../../frontend/src/lib/graphLayout.test.ts) | non-determinism, NaN coordinates, a layout that collapses to a point |
| [`webgpu/gnn.test.ts`](../../frontend/src/webgpu/gnn.test.ts) | **a wrong backward pass**, by finite differences, for every architecture at 1–3 layers, with dropout and without |
| [`webgpu/gnnRuntime.test.ts`](../../frontend/src/webgpu/gnnRuntime.test.ts) | a malformed graph reaching the GPU, before any device work |
| [`webgpu/gat.test.ts`](../../frontend/src/webgpu/gat.test.ts) | attention that is not attention — the gradient check pins the *derivative*, this pins the forward pass's convex-combination property |
| [`webgpu/graphSession.test.ts`](../../frontend/src/webgpu/graphSession.test.ts) | a second layout on a second load, a cancel surfacing as a failure, the feature matrix escaping the worker |
| [`lib/random.test.ts`](../../frontend/src/lib/random.test.ts) | a stream that moves between runs, and Box–Muller's `log(0)` |
| [`hooks/useGraphTraining.test.ts`](../../frontend/src/hooks/useGraphTraining.test.ts) | a re-entrant load, a depth point stacking instead of replacing, a leaked worker |
| [`components/graph/GraphCanvas.test.tsx`](../../frontend/src/components/graph/GraphCanvas.test.tsx) | an off-centre or stretched drawing, via the pure coordinate mapping |
| [`__tests__/routes/graph.test.tsx`](../../frontend/src/__tests__/routes/graph.test.tsx) | the page contract: four slots, nothing runs on mount, a control that shouldn't run doesn't |
| [`e2e/specs/webgpu/graph.spec.ts`](../../frontend/e2e/specs/webgpu/graph.spec.ts) | the WGSL kernel disagreeing with the CPU reference, and — `@slow` — a real training run pinned by an **accuracy floor** |

Three notes on the gradient check, because it is the load-bearing test:

- **It is checked at a generic point, not the initial one.** Biases initialise to zero, so
  a node whose previous layer is entirely dead has a preactivation of *exactly* 0 — sitting
  on ReLU's kink, where the function is not differentiable and a central difference reports
  half the true gradient. That is a property of the test point, not of the backward pass.
- **Dropout needs its own pass.** The masked input transpose is reachable only with dropout
  on, so a wrong `nzT` mapping is invisible to the plain check. Verified: corrupting it
  fails the six dropout cases and passes all the others.
- **It has teeth.** Dropping the self-loop from the shader moves the kernel cross-check's
  worst-case error from under 1e-5 to 0.87.

Two more, from the pass that filled the gaps above:

- **A gradient check does not pin a forward pass.** A softmax normalised over the
  wrong set still differentiates consistently, so GAT's finite-difference checks would
  pass while its attention weights were not attention weights. `gat.test.ts` asserts the
  property instead — every output is a convex combination of its closed neighbourhood —
  and calibrates it by zeroing the attention vectors, which must collapse the layer to an
  exact mean.
- **Extract the geometry from a canvas.** happy-dom gives a canvas no 2D context and has
  no `ResizeObserver`, so a component test of `GraphCanvas` reaches only the guarded
  early-returns. Pulling `layoutToPixels` out as a pure function immediately surfaced an
  off-centre drawing — the span was centred and then shifted again by half the padding,
  giving a 12px gap on one side and 4px on the other.

Unlike the audio and vision routes, the unit suite here **can** catch a broken model:
there is no network and no ONNX session to mock away. The E2E specs still exist for the
one thing it cannot do, which is execute WGSL.

---

## 7. Reference

- **[`src/webgpu/`](../../frontend/src/webgpu/)** — `getGPUDevice`, `createComputePipeline`,
  `readBackFloat32`, the matmul and elementwise shaders, and `worker.ts`. This route is a
  direct extension of that path, not of the Transformers.js one, and the two never mix in
  one directory.
- **`detectWebGPU()`** ([`webgpu/capabilities.ts`](../../frontend/src/webgpu/capabilities.ts))
  matters more here than anywhere else, because there is no WASM runtime to fall back to.
  It never throws. Note the common false negative: `navigator.gpu` needs a **secure
  context**, so a plain-HTTP LAN origin hides WebGPU entirely, and Firefox needs
  `dom.webgpu.enabled`. The test harness had exactly this bug — it probed from
  `about:blank`, whose opaque origin is not secure, so every spec in `e2e/specs/webgpu/`
  skipped itself on every machine. See
  [`webgpu-inference.md`](../explanations/webgpu-inference.md).
- **Datasets**: Cora, bundled. `graphs-datasets/PROTEINS` for §3.4, to be fetched at runtime.
- **Page construction**: [`adding-a-task-page.md`](../guides/adding-a-task-page.md) and
  [`model-visualization.md`](../standards/model-visualization.md).
