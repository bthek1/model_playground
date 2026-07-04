// Element-wise binary op over two equally-shaped, flattened tensors.
// `op` selects the arithmetic: 0=add, 1=sub, 2=mul (Hadamard), 3=div.
// One kernel covers every same-shape binary op — dispatch over the length.

struct Params {
  op: u32,
  len: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) {
    return;
  }

  let x = a[i];
  let y = b[i];
  var r = 0.0;
  switch params.op {
    case 0u: { r = x + y; }
    case 1u: { r = x - y; }
    case 2u: { r = x * y; }
    case 3u: { r = x / y; }
    default: { r = 0.0; }
  }
  out[i] = r;
}
