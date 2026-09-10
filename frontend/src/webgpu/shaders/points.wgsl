// Point-cloud render pass — the first *render* shader in this runtime; every
// other WGSL file here is compute.
//
// **Instanced quads, not `point-list`.** WebGPU's point primitive is always
// exactly one pixel: there is no `gl_PointSize`. On a 1200x800 backing store a
// few hundred thousand one-pixel points read as faint noise rather than as a
// scene, which would make the page look like the model failed. So each point is
// an instance, six vertices form two triangles, and the corner offsets are
// applied in clip space — scaled by `w` so a point keeps the same size on
// screen whether it is near or far.
//
// The camera is a plain orbit built here rather than on the CPU: dragging the
// view then rewrites 48 bytes of uniform instead of re-uploading a
// multi-megabyte vertex buffer, which is the difference between an orbit that
// tracks the pointer and one that stutters.

struct Camera {
  yaw: f32,
  pitch: f32,
  distance: f32,
  aspect: f32,
  // Centre of the cloud, so orbiting turns around the scene, not the origin.
  center: vec3<f32>,
  _pad0: f32,
  // Half-size of a point in clip units, per axis — the CPU folds the canvas
  // resolution in, so the shader needs no resolution of its own.
  pointRadius: vec2<f32>,
  _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

// Two triangles, counter-clockwise, covering [-1, 1]^2.
const CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(
  @builtin(vertex_index) vertex: u32,
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
) -> VertexOut {
  // Orbit: translate the cloud to the origin, rotate by yaw then pitch, then
  // push it away from the eye by `distance`.
  let p = position - camera.center;

  let cy = cos(camera.yaw);
  let sy = sin(camera.yaw);
  let rotatedY = vec3<f32>(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);

  let cp = cos(camera.pitch);
  let sp = sin(camera.pitch);
  let view = vec3<f32>(
    rotatedY.x,
    cp * rotatedY.y - sp * rotatedY.z,
    sp * rotatedY.y + cp * rotatedY.z,
  );

  // The image's +y runs downwards (row 0 is the top) and clip space's runs up,
  // so y is negated here. Without it the cloud renders upside down — which on a
  // roughly symmetric scene is easy to mistake for a correct render.
  let eye = vec3<f32>(view.x, -view.y, view.z + camera.distance);

  // Perspective, hand-rolled: 60° vertical field of view, near 0.05, far 200.
  let f = 1.0 / tan(0.5236);
  let near = 0.05;
  let far = 200.0;

  var clip = vec4<f32>(
    (f / camera.aspect) * eye.x,
    f * eye.y,
    (far + near) / (near - far) * eye.z + (2.0 * far * near) / (near - far),
    -eye.z,
  );

  // Expand the instance into a screen-facing quad. Multiplying by `clip.w`
  // cancels the perspective divide, so points stay a constant size on screen
  // instead of collapsing to nothing in the distance.
  let corner = CORNERS[vertex % 6u];
  clip = vec4<f32>(
    clip.x + corner.x * camera.pointRadius.x * clip.w,
    clip.y + corner.y * camera.pointRadius.y * clip.w,
    clip.z,
    clip.w,
  );

  var out: VertexOut;
  out.position = clip;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
