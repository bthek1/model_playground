// MNIST dataset loading for in-browser training. The dataset is fetched from a
// CDN as a sprite PNG (one 28×28 image flattened per row) plus a uint8 one-hot
// label file — the well-known tfjs "learnjs" hosting. Pixels are decoded via a
// canvas and normalised to [0,1]; labels are collapsed to integer classes.
//
// Only the pure transforms (normalizePixels, labelsFromOnehot, sliceDataset) are
// unit-tested; the fetch + canvas decode needs a browser and is exercised
// manually.

export const IMAGE_SIZE = 784; // 28 * 28
export const NUM_CLASSES = 10;
export const MNIST_POOL_SIZE = 65000; // images available in the sprite

const SPRITE_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const LABELS_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

// The sprite is tall; decode it in horizontal strips so the backing canvas
// stays a sane size.
const DECODE_CHUNK_ROWS = 5000;

export interface MnistPool {
  /** `count * IMAGE_SIZE` pixel values in [0,1], row-major per image. */
  images: Float32Array;
  /** `count` integer class labels in [0,9]. */
  labels: Uint8Array;
  count: number;
}

export interface MnistSplit {
  xTrain: Float32Array;
  yTrain: Uint8Array;
  xTest: Float32Array;
  yTest: Uint8Array;
  trainSize: number;
  testSize: number;
}

/** Extract the red channel of RGBA bytes and scale to [0,1] (MNIST is grey). */
export function normalizePixels(rgba: Uint8Array | Uint8ClampedArray): Float32Array {
  const out = new Float32Array(rgba.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4] / 255;
  return out;
}

/** Collapse a flat `count × numClasses` one-hot buffer to integer labels. */
export function labelsFromOnehot(
  onehot: Uint8Array,
  count: number,
  numClasses = NUM_CLASSES,
): Uint8Array {
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    let cls = 0;
    for (let c = 0; c < numClasses; c++) {
      if (onehot[i * numClasses + c]) {
        cls = c;
        break;
      }
    }
    out[i] = cls;
  }
  return out;
}

/**
 * Split a loaded pool into disjoint train/test sets: `trainSize` images from the
 * front, `testSize` from the block immediately after. Throws if the pool is too
 * small. Returned buffers are fresh copies (safe to transfer to a worker).
 */
export function sliceDataset(
  pool: MnistPool,
  trainSize: number,
  testSize: number,
): MnistSplit {
  if (trainSize + testSize > pool.count) {
    throw new Error(
      `Requested ${trainSize}+${testSize} images but only ${pool.count} are loaded.`,
    );
  }
  const D = IMAGE_SIZE;
  const xTrain = pool.images.slice(0, trainSize * D);
  const yTrain = pool.labels.slice(0, trainSize);
  const xTest = pool.images.slice(trainSize * D, (trainSize + testSize) * D);
  const yTest = pool.labels.slice(trainSize, trainSize + testSize);
  return { xTrain, yTrain, xTest, yTest, trainSize, testSize };
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error(`Could not load the MNIST sprite from ${url}.`));
    img.src = url;
  });
}

/**
 * Fetch and decode the first `count` MNIST images (+ labels) in the browser.
 * `onProgress` receives a 0→1 fraction as strips are decoded.
 */
export async function loadMnistPool(
  count: number,
  onProgress?: (fraction: number) => void,
): Promise<MnistPool> {
  if (count > MNIST_POOL_SIZE) {
    throw new Error(`MNIST has ${MNIST_POOL_SIZE} images; requested ${count}.`);
  }

  const [img, labelsResponse] = await Promise.all([
    loadImage(SPRITE_URL),
    fetch(LABELS_URL),
  ]);
  if (!labelsResponse.ok) {
    throw new Error(`Could not load MNIST labels (HTTP ${labelsResponse.status}).`);
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not get a 2D canvas context to decode MNIST.");
  canvas.width = img.width; // == IMAGE_SIZE

  const images = new Float32Array(count * IMAGE_SIZE);
  for (let start = 0; start < count; start += DECODE_CHUNK_ROWS) {
    const rows = Math.min(DECODE_CHUNK_ROWS, count - start);
    canvas.height = rows;
    ctx.drawImage(img, 0, start, img.width, rows, 0, 0, img.width, rows);
    const { data } = ctx.getImageData(0, 0, img.width, rows);
    const strip = normalizePixels(data);
    images.set(strip, start * IMAGE_SIZE);
    onProgress?.((start + rows) / count);
  }

  const onehot = new Uint8Array(await labelsResponse.arrayBuffer());
  const labels = labelsFromOnehot(onehot, count);

  return { images, labels, count };
}
