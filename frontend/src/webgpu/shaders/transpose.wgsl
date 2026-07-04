// Transpose a row-major (rows x cols) matrix into a (cols x rows) matrix.
// Each invocation copies one source element to its transposed position.

struct Dims {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= dims.rows || col >= dims.cols) {
    return;
  }
  // input[row, col] (rows x cols) -> output[col, row] (cols x rows)
  output[col * dims.rows + row] = input[row * dims.cols + col];
}
