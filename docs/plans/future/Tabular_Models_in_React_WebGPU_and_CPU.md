# Tabular Models in the Browser (WebGPU or CPU)

> The **Tabular** category of `components/layout/taskTaxonomy.ts`, task by task:
> what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly). No Python
> server in the inference path.

**Nothing in this category is built** — Tabular Classification, Tabular
Regression and Time Series Forecasting all render the `/tasks/$slug` placeholder.
But this is the category with the shortest distance to a working page, because
the closest existing route, `/training`, already does most of it.

This category is the exception to the pattern the rest of the app follows, and
the exception is good news. There are almost no Hugging Face checkpoints here:
the reference implementations are scikit-learn, XGBoost, LightGBM and CatBoost,
and the two neural models worth wanting — TimesFM and PatchTST — have no ONNX
export at all.

That sounds like a category with no browser story. It is the opposite. **Tabular
models are the only models here small enough to train in the browser from
scratch**, in a few seconds, on data the user pastes in. A gradient-boosted tree
on ten thousand rows is kilobytes of parameters, and
[`src/webgpu/`](../../../frontend/src/webgpu/) already has the machinery: the
[`/training`](../../../frontend/src/routes/training.tsx) route trains a linear
model in hand-written WGSL and visualises it converging.

So this category does not produce inference pages. It produces **training
pages**, and they are the ones where the user's own CSV never leaves the device.
That is a stronger privacy argument than any pretrained route can make, because
tabular data is the kind people actually mind about: payroll, patients, sales.

---

## 1. The core stack

Two options, and unusually the second one is often the right answer.

```bash
# A. Run a model that was trained in Python, exported to ONNX
npm i onnxruntime-web
npm i skl2onnx            # (python side) sklearn -> ONNX
npm i onnxmltools         # (python side) XGBoost / LightGBM -> ONNX

# B. Train in the browser: no ML library at all
#    Plain TypeScript for trees, WGSL compute shaders for the linear models
```

There is no Transformers.js in this category. Nothing here is a transformer.

### Option A: export from Python, infer in the tab

`skl2onnx` and `onnxmltools` cover everything scikit-learn and the three
production boosters produce:

```python
# Python side, after fitting.
from skl2onnx import to_onnx
onx = to_onnx(model, X_train[:1].astype(np.float32), target_opset=17)
Path("../../datasets/exports/histgb.onnx").write_bytes(onx.SerializeToString())
```

```ts
// In the browser
import * as ort from "onnxruntime-web";
const session = await ort.InferenceSession.create("/models/histgb.onnx");
const input = new ort.Tensor("float32", flatRows, [nRows, nFeatures]);
const { label, probabilities } = await session.run({ float_input: input });
```

A HistGradientBoosting model on twenty features is typically well under a
megabyte, so the LOAD state is nearly instant and the large-model warning never
appears. That changes the page: for once, SELECT and LOAD collapse to almost
nothing and the workbench is the whole screen.

**Use `wasm`, not `webgpu`, for tree ensembles.** ONNX Runtime's tree operators
run on CPU regardless, and a batch of a few thousand rows is microseconds
either way. WebGPU earns its place on the linear and neural models below, not
on trees.

### Option B: train in the browser

The genuinely interesting path, and the one that fits the existing `/training`
route. Three tiers, in increasing order of work:

| Model | Implementation | Where the compute goes |
|---|---|---|
| Ridge / logistic regression | closed form or gradient descent | WGSL, a matmul and a reduction. `webgpu/shaders/` already has both |
| Decision tree / random forest | plain TypeScript, recursive splits | main thread is too slow, so a Web Worker |
| Gradient boosting | plain TypeScript over the tree above | Web Worker, and cap the depth |

The linear case is a direct extension of what `webgpu/worker.ts` already does.
The tree case is a Worker with no GPU involvement at all, which is worth stating
plainly on the page: **not everything belongs on a GPU**, and a tabular page
that says so is more honest than one that pretends.

---

## 2. The shape of a training page

The four-slot pattern still holds, with one substitution. There are no weights
to download, so LOAD becomes **FIT**:

```
SELECT   the model family and its hyperparameters
FIT      run training, with a live loss or a live tree count
RUN      predict on a held-out split, or on a row the user types
OUTPUT   the metric, the confusion matrix, the residual plot, the feature importances
```

Machine A is reused **unchanged** — `idle → loading → ready | error`, with
`progress` as a self-loop — and the LOAD slot is simply relabelled by the route.
Do not invent a `fitting` status: `useModelWorker` owns the enum, and a category
that renames it costs the "one vocabulary" property that makes the other pages
readable. `model/progress.ts` is written around a byte count, so a fitting page
uses its **indeterminate** mode plus its own iteration counter.

`/training` is the precedent for this whole substitution, and it is the
documented exception in
[`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
§7: it keeps its own full-bleed canvas layout rather than the four bands. A CSV
page should *not* claim that exemption — it has a genuine input surface and a
genuine result, so it uses `ModelPage` like everything else.

Two things that belong in OUTPUT rather than in prose:

- **The confusion matrix** and **the residual plot** exist because the aggregate
  number hides the failure. In a notebook they are a chart; on a page they are the
  main panel. Use the lazy
  [`components/charts/EChart.tsx`](../../../frontend/src/components/charts/EChart.tsx)
  wrapper — `echarts` is heavy and must stay code-split.
- **Feature importances.** A bar chart per column, and it is the output
  non-technical users actually read.

The CSV parse belongs in the worker, not the component. A ten megabyte file
parsed on the main thread freezes the tab for a second, and the user's first
interaction with the page is a stutter.

---

## 3. Task by task

### 3.1 Tabular Classification, the best training page in the repo

Taxonomy task **Tabular Classification** · not built. Reference ladder: logistic
regression, random forest, HistGradientBoosting, MLP, XGBoost, LightGBM, CatBoost.

Everything in this ladder trains in the browser in seconds on a few thousand
rows. There is no checkpoint, so there is nothing to verify against the Hub and
nothing to download — which also means no catalogue entry and no
`just fe-e2e-models` row.

The page writes itself: the user drops a CSV, picks a target column, and the page
fits the whole ladder.
Logistic regression as the floor, random forest as the zero-tuning baseline,
gradient boosting as the workhorse, and an MLP as the deep baseline that does
not win. **Ship the MLP precisely because it loses.** That result is the point
the point of the whole ladder, and it is the kind of thing people do not believe
until they watch it happen on their own data.

Two things that must survive the port:

- **Calibration and the threshold.** The default 0.5 is a convention, not a
  decision. A slider that moves the threshold and updates
  precision, recall and the confusion matrix live is the single most useful
  control in this category.
- **Which columns actually matter.** Permutation importance is a
  loop over columns with a shuffle in the middle, so it is a few lines and it
  runs fast enough to be interactive.

### 3.2 Tabular Regression, the same page with different diagnostics

Taxonomy task **Tabular Regression** · not built. Reference ladder: ridge, random
forest, HistGradientBoosting, quantile regression, XGBoost, LightGBM, CatBoost.

Same worker, same fit loop, different output panel. Three things differ enough
to matter:

- **Quantile regression** ships an interval instead of a number,
  and an interval is far better to render than a point estimate. Draw the
  prediction band; it is more honest and it looks better.
- **The log-transform trap** is a toggle. Fit on the raw target and
  on `log1p(target)`, then show that the error metrics are not comparable
  between them. This is a mistake people make constantly, and one toggle
  demonstrates it.
- **Residuals** replace the confusion matrix. Predicted against
  actual, with the residual on the second axis.

Ridge is the model to implement in WGSL, since it has a closed form and the
whole solve is one matmul, one small inverse and one more matmul. That makes it
the natural companion to the existing `/training` route rather than a new kind
of page.

### 3.3 Time Series Forecasting, baselines in the browser, models on the server

Taxonomy task **Time Series Forecasting** · not built. Reference: naive
baselines, TimesFM 2.0, PatchTST.

This one splits.

| Component | Browser | Notes |
|---|---|---|
| The baselines you have to beat | Yes, trivially | naive, seasonal naive and drift are arithmetic. Twenty lines |
| Backtesting | Yes | a loop over rolling splits. No model needed to demonstrate the idea |
| `google/timesfm-2.0-500m-pytorch` | No | **checked: no ONNX weights in the repo** |
| `ibm-granite/granite-timeseries-patchtst` | No | **checked: no ONNX weights in the repo** |

Which leaves an honest and still worthwhile page: the user pastes a series, the
page draws the baselines, and the backtest shows why a single train/test split is
not an evaluation. One split lies, and a page that lets the user watch the metric
swing across rolling windows makes that unarguable.

If a foundation forecaster is genuinely wanted in the browser, the path is to
export PatchTST yourself with `torch.onnx.export` and host the artefact, then
serve it through `ModelCard.weights_url` like any other checkpoint. It is a small
encoder, so it would work. That is a project rather than a page — it gets its own
plan in [`../in-progress/`](../in-progress/).

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | How | Best backend | If not |
|---|---|---|---|---|
| **Tabular Classification** | Yes, train and infer | TypeScript trees in a Worker, or an ONNX export from Python | WASM for trees, WebGPU for linear | - |
| **Tabular Regression** | Yes, train and infer | as above, plus a closed-form ridge in WGSL | WASM for trees, WebGPU for ridge | - |
| **Time Series Forecasting**, baselines and backtesting | Yes | plain arithmetic, and a rolling loop | main thread is fine | - |
| **Time Series Forecasting**, TimesFM 2.0 / PatchTST | No | no ONNX export in either repo | - | server API, or export it yourself |

Rule of thumb: **this category inverts the usual question.** Everywhere else the
question is whether the weights fit in a tab. Here the weights are trivial and
the question is whether the *training* fits, and for a few thousand rows it
comfortably does.

---

## 5. Memory and performance notes

Different constraints from every other category, because the data is the load
rather than the model.

- **The dataset is the memory budget.** A million-row CSV as JavaScript objects
  is hundreds of megabytes. Parse into typed arrays, one `Float32Array` per
  column, and never build an array of row objects.
- **Cap the row count in the UI**, with an explicit message. Sampling 50000 rows
  from a large file and saying so is better than freezing on the full file.
- **Trees go in a Worker, not on the GPU.** Recursive splitting is branch-heavy
  and does not vectorise. Attempting it in WGSL is the wrong instinct, and the
  page should say why.
- **Linear models go on the GPU**, where a matmul is what the hardware is for.
  This is the honest split, and demonstrating it is itself a teaching point.
- **Report progress by iteration**, not by bytes. `model/progress.ts` is written
  around a byte count, so a fitting page uses its indeterminate mode plus its own
  iteration counter.
- **There is nothing to dispose.** No GPU session outlives the fit, which makes
  this the one category where the one-model-live rule does not apply. The GPU
  device itself is memoised and shared (`webgpu/device.ts`) — do not destroy it.
- **`DeviceStatus`, not `ModelStatus`, in the LOAD slot** for the WGSL half.
  There are no weights to download; the question the page actually raises is
  whether there is a GPU at all, and
  [`DeviceStatus`](../../../frontend/src/components/model/DeviceStatus.tsx)
  answers exactly that. `/tensor` is the precedent.

---

## 6. Reference

- **onnxruntime-web** for models exported from Python. `wasm` provider for trees,
  `webgpu` for anything dense. Already a dependency; import it as
  `onnxruntime-web/webgpu` so Vite emits one shared WASM asset rather than two.
- **skl2onnx** and **onnxmltools** on the Python side, to produce those exports.
- **[`src/webgpu/`](../../../frontend/src/webgpu/)**: the existing WGSL matmul,
  transpose, scale and elementwise shaders plus `linearModel.ts` are the
  foundation for the linear models here. `/training` and `/tensor` are the two
  routes to read first.
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
- Model and method recommendations here come from a companion collection of
  Python notebooks, which is a separate project and not a dependency of this
  repo. Unusually, almost nothing in this category needs a checkpoint at all.
- **In-repo standards**:
  [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
  §7 (where the pattern bends for compile-only and training pages),
  [`../../standards/model-visualization.md`](../../standards/model-visualization.md).
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
