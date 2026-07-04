// Multiply every element of a flattened tensor by a scalar. The uniform packs
// a u32 length and an f32 scalar into 8 bytes (see runTensorOp for the layout).

struct Params {
  len: u32,
  scalar: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) {
    return;
  }
  out[i] = a[i] * params.scalar;
}
