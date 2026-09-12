// Message passing as a scaled gather over a compressed-sparse-row graph:
//
//     out[i] = α_i · Σ_{j ∈ N(i) ∪ {i}} β_j · x[j]
//
// One invocation per node, each walking its own neighbour slice. Every GNN
// architecture in the playground except GAT is a choice of the two scale
// vectors — GCN's symmetric 1/√(deg+1) on both sides, GraphSAGE's mean on one,
// GIN's plain sum — and the transpose the backward pass needs is the same
// dispatch with α and β swapped. See webgpu/gnn.ts for the derivation.
//
// Two invariants this shader depends on, both asserted in lib/cora.test.ts:
//
//   - `colIdx` holds **no self-loops**. The `acc` below is seeded with the
//     node's own features, which is the `+ I` in `A_hat = A + I`. A stored
//     self-loop would count it twice, and the result would still look like a
//     working model.
//   - the graph is **symmetric**, which is what makes the swapped-scale
//     dispatch a real transpose rather than an approximation of one.
//
// CSR and not a dense adjacency matrix: Cora as a dense 2708 x 2708 f32 matrix
// is 29 MB of mostly zeros, against about 40 KB as CSR. On any graph worth
// drawing the dense form is the wrong shape long before it is slow.

struct Dims {
  nNodes: u32,
  nFeat: u32,
};

@group(0) @binding(0) var<uniform>             dims   : Dims;
@group(0) @binding(1) var<storage, read>       rowPtr : array<u32>;
@group(0) @binding(2) var<storage, read>       colIdx : array<u32>;
@group(0) @binding(3) var<storage, read>       alpha  : array<f32>;
@group(0) @binding(4) var<storage, read>       beta   : array<f32>;
@group(0) @binding(5) var<storage, read>       x      : array<f32>;
@group(0) @binding(6) var<storage, read_write> out    : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let node = gid.x;
  if (node >= dims.nNodes) {
    return;
  }

  let start = rowPtr[node];
  let end = rowPtr[node + 1u];
  let base = node * dims.nFeat;
  let selfScale = beta[node];
  let outScale = alpha[node];

  // Feature-outer with a register accumulator: `out` is written exactly once per
  // element, and the neighbour indices — at most a few per node on a citation
  // graph — stay hot in cache across the feature loop.
  for (var f = 0u; f < dims.nFeat; f = f + 1u) {
    var acc = selfScale * x[base + f];
    for (var e = start; e < end; e = e + 1u) {
      let j = colIdx[e];
      acc = acc + beta[j] * x[j * dims.nFeat + f];
    }
    out[base + f] = outScale * acc;
  }
}
