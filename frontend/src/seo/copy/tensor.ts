import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Basic matrix and tensor operations, computed on your GPU via raw WebGPU in a Web Worker.",
  about: [
    "Tensor arithmetic is the handful of operations every neural network is built from. This page offers seven of them on two-dimensional matrices: element-wise addition, subtraction, division and multiplication (the Hadamard product), matrix multiplication, matrix transpose, and multiplying every element by a scalar. Type the operands as text, one row per line, and the page shows each operand's shape as you edit it.",
    "Each operation is a hand-written WGSL compute shader, with no machine learning framework in between. The four element-wise operations share one kernel, matmul reuses the app's reference matrix multiplication kernel, and transpose and scale each have a small kernel of their own. They run in a Web Worker, so the page never freezes while the GPU works, and the numbers are 32-bit floats.",
    "Shapes are checked on the CPU before anything reaches the GPU: element-wise operations need operands of identical shape, and matrix multiplication needs the inner dimensions to agree, so a mismatch comes back as a plain sentence rather than a driver error. The result arrives with its shape and the time the GPU spent on it, drawn as a diverging heatmap — red above zero, blue below, stronger colour for a larger magnitude — with the raw values in a grid underneath.",
    "There is nothing to download: the second band only checks that the browser can open a WebGPU device. Without one the page explains why and cannot compute, because it has no CPU path. It also stops at matrices — there is no broadcasting and no tensor of higher rank.",
  ],
  aliases: [
    "tensor arithmetic",
    "matrix multiplication",
    "element-wise operations",
    "Hadamard product",
    "matrix transpose",
    "WGSL compute shader",
  ],
};
