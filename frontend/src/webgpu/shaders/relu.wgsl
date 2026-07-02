// Element-wise ReLU activation, in place. A second tiny kernel to show the
// storage-buffer + dispatch pattern for pointwise ops (activations, adds,
// normalisation). Bind a single read_write buffer and dispatch over its length.

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&data)) {
    return;
  }
  data[i] = max(data[i], 0.0);
}
