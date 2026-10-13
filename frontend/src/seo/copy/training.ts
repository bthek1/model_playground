import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Train a handwritten digit classifier on MNIST in your browser, and watch its weights turn into digits.",
  about: [
    "MNIST is the classic dataset of handwritten digits: 28×28 greyscale images, each labelled 0 to 9. This page fits the simplest model that can read them — softmax regression, also called multinomial logistic regression — with one weight per pixel per class, 784 → 10, or 7,850 parameters including the biases. Choose how many training and test images to use, set the learning rate, batch size and number of epochs, then press Start.",
    "Training is mini-batch gradient descent. The two large matrix multiplications in every step, the forward logits and the weight gradient, run as WGSL compute shaders on WebGPU inside a Web Worker, so the page stays responsive; the softmax, the cross-entropy loss and the update are plain arithmetic. The images come from a public CDN, are decoded in the browser and kept in IndexedDB, so a reload does not fetch them again. Nothing you train is uploaded.",
    "Behind the controls, each class's weights are drawn as a 28×28 image — red where a pixel argues for the digit, blue where it argues against — and they sharpen into rough digit shapes as the epochs pass. Charts show the loss per batch and the train and test accuracy after each epoch.",
    "It is deliberately a single linear layer, with no hidden layers and no convolutions, so every parameter is on screen at once. It needs a WebGPU device: there is no CPU fallback, and without one the page says why and leaves Start disabled.",
  ],
  aliases: [
    "MNIST",
    "handwritten digit classifier",
    "softmax regression",
    "multinomial logistic regression",
    "mini-batch gradient descent",
    "WebGPU",
  ],
};
