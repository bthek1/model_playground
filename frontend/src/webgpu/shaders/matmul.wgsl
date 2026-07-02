// Naive row-major matrix multiply: C(MxN) = A(MxK) * B(KxN).
//
// This is the smallest useful building block for neural-net layers (a dense
// layer is a matmul + bias + activation). It is intentionally simple and
// correct rather than maximally fast — start here, then specialise (tiling,
// shared memory, quantized weights) per model. See docs/guides/adding-a-model.md.

struct Dims {
  m: u32,
  k: u32,
  n: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.m || col >= dims.n) {
    return;
  }

  var acc = 0.0;
  for (var i: u32 = 0u; i < dims.k; i = i + 1u) {
    acc = acc + a[row * dims.k + i] * b[i * dims.n + col];
  }
  c[row * dims.n + col] = acc;
}
