# Tabular Models in the Browser (WebGPU or CPU)

> The **Tabular** category of
> [`taskTaxonomy.ts`](../../frontend/src/components/layout/taskTaxonomy.ts), task by task:
> what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly). No Python server
> in the inference path — and, in this category, no checkpoint either.
>
> This file began as issue [#7](https://github.com/bthek1/model_playground/issues/7) and
> moved here when its first route shipped, the same way
> [`audio.md`](audio.md), [`vision.md`](vision.md) and [`graph.md`](graph.md) did: a
> roadmap that documents *shipped* code has to be reviewable in the same pull request as
> the code it describes, which an issue body cannot be.

| Roadmap section | Route | Status |
|---|---|---|
| Tabular Classification (§3.1) | [`/tabular-classification`](../../frontend/src/routes/tabular-classification.tsx) | **Shipped** — the four-family ladder, fitted in the tab |
| Tabular Regression (§3.2) | [`/tabular-regression`](../../frontend/src/routes/tabular-regression.tsx) | **Shipped** — closed-form ridge, quantile bands, the log-transform trap |
| Time Series Forecasting (§3.3) | [`/time-series-forecasting`](../../frontend/src/routes/time-series-forecasting.tsx) | **Shipped as baselines + backtesting** — the two foundation models have no ONNX export |

**The category is complete as scoped: 3 of 3.** The build-out is recorded in the closed
plan issues [#48](https://github.com/bthek1/model_playground/issues/48) (`src/tabular/` +
§3.1), [#49](https://github.com/bthek1/model_playground/issues/49) (§3.2) and
[#50](https://github.com/bthek1/model_playground/issues/50) (§3.3).

This category is the exception to the pattern the rest of the app follows, and the
exception is good news. There are almost no Hugging Face checkpoints here: the reference
implementations are scikit-learn, XGBoost, LightGBM and CatBoost, and the two neural models
worth wanting — TimesFM and PatchTST — have no ONNX export at all.

That sounds like a category with no browser story. It is the opposite. **Tabular models
are the only models here small enough to train in the browser from scratch**, in a few
seconds, on data the user drops in. A gradient-boosted tree on ten thousand rows is
kilobytes of parameters, and [`src/webgpu/`](../../frontend/src/webgpu/) already has the
machinery: [`/training`](../../frontend/src/routes/training.tsx) trains a linear model in
hand-written WGSL and visualises it converging.

So this category does not produce inference pages. It produces **training pages**, and
they are the ones where the user's own CSV never leaves the device. That is a stronger
privacy argument than any pretrained route can make, because tabular data is the kind
people actually mind about: payroll, patients, sales.

---

## The feasibility bar this file is filtered by

Every row here clears the same two tests the other category roadmaps are swept against:

1. **It runs client-side** — on WebGPU where the maths is a matmul or a reduction, on the
   CPU where the CPU is the right answer. Decision trees are recursive splits with no
   linear algebra in them, so they are TypeScript in a Worker; a ridge regression is a
   closed form and belongs in WGSL. Insisting on the GPU for the trees would make them
   slower, not more legitimate. What is *not* acceptable is a row needing a server.
2. **Its cheapest usable checkpoint is under ~500 MB**, measured off the Hub.

**This category passes both, and the second one vacuously**: §3.1 and §3.2 download no
checkpoint at all, and §3.3 downloads nothing whatsoever.

So the bar had to be replaced with a real one, and the replacement is §0 below.

---

## 0. The bar that actually applies: does the *fit* fit?

Measured before the pages were designed around it, because the answer could have cut a
family off the ladder. Synthetic data, 20 columns, two classes, fitted in Node with the
**CPU-reference matmul** — so the two GPU families are upper bounds, and the trees, which
never touch the GPU, are what a browser actually sees.

| rows | binning | forest 60×d8 | boosting 120×d4 | boosting 120×d6 | logistic 30ep | MLP 40ep |
|---|---|---|---|---|---|---|
| 10 000 | 21 ms | 1.0 s | 1.7 s | **3.0 s** | 0.14 s | 1.8 s |
| 50 000 | 106 ms | 4.6 s | 7.7 s | **12.2 s** | 0.71 s | 9.2 s |
| 200 000 | 467 ms | 16.9 s | 31.4 s | **51.0 s** | 3.20 s | 38.0 s |

Three conclusions, all of them now constants in
[`tabular/limits.ts`](../../frontend/src/tabular/limits.ts) with the table beside them:

1. **Nothing is cut.** Every family fits 10 000 rows in three seconds or less, so the
   ladder ships whole — including the MLP, which is on it precisely because it loses on
   *accuracy*, not on time.
2. **`MAX_ROWS` is 50 000, and gradient boosting at depth 6 set it.** At 200 000 rows that
   configuration is 51 seconds, which no progress bar and Stop button make honest; at
   50 000 the worst case on the whole ladder is 12 seconds.
3. **Boosting's depth is capped at 6**, from the same table: depth 6 costs 1.6× depth 4 at
   every size, and depth 8 would put the worst case back over twenty seconds at the row
   cap. The forest keeps a cap of 12, because a bagged tree is fitted once rather than once
   per boosting round per class.

**The rule of thumb this category inverts:** everywhere else the question is whether the
weights fit in a tab. Here the weights are trivial and the question is whether the
*training* fits — and for a few tens of thousands of rows it comfortably does.

---

## 1. The core stack

Two options, and unusually the second one is the right answer.

```bash
# A. Run a model trained in Python and exported to ONNX
npm i onnxruntime-web        # already a dependency
npm i skl2onnx onnxmltools   # (python side)

# B. Train in the browser: no ML library at all
#    Plain TypeScript for trees, WGSL compute shaders for the linear models
```

There is no Transformers.js in this category. Nothing here is a transformer.

### Option A was scoped out, deliberately

`skl2onnx` and `onnxmltools` cover everything scikit-learn and the three production
boosters produce, and a HistGradientBoosting model on twenty features is well under a
megabyte — so the LOAD state would be nearly instant and the large-model warning would
never appear.

It is still out of scope, and the reason is not size: it needs a **Python step and an
artefact to host**, which is a project rather than a page. Nothing about the shipped
routes unblocks it, and nothing about it unblocks them. If a comparison against a
Python-fitted export is wanted later it gets its own plan. (The inverse — *exporting* a
model fitted on these pages, as ONNX or as JSON coefficients — is also useful and also a
separate plan.)

### Option B: train in the browser

Three tiers, and where each one's compute goes:

| Model | Implementation | Where the compute goes |
|---|---|---|
| Ridge / logistic regression | closed form or gradient descent | WGSL: a matmul and a reduction, through the `MatmulFn` seam |
| Decision tree / random forest | plain TypeScript, recursive splits | a Web Worker, with **no GPU involvement at all** |
| Gradient boosting | plain TypeScript over the tree above | a Web Worker, and the depth is capped |

**The tree case is worth stating on the page, not just here: not everything belongs on a
GPU.** Recursive splitting is branch-heavy and does not vectorise — every node asks a
different question of a different subset of rows, which is the opposite of what a compute
shader is for. Attempting it in WGSL is the wrong instinct, and a tabular page that says so
is more honest than one that pretends. Both shipped routes render that sentence beside the
family the user picked.

---

## 2. The shape of a training page

The four-slot pattern holds, with one substitution. There are no weights to download, so
LOAD becomes **FIT**:

```
SELECT   the model family and its hyperparameters
FIT      run training, with a live loss or a live tree count
RUN      the CSV, the target column, and a row to predict
OUTPUT   the metric, the confusion matrix or residual plot, the feature importances
```

Machine A is reused **unchanged** — `idle → loading → ready | error` — and the LOAD slot is
relabelled by the route. Do not invent a `fitting` status: `useModelWorker` owns the enum,
and a category that renames it costs the one vocabulary that makes the other pages readable.

**Two corrections this file carries, both made while planning and both the useful part:**

- **The original §2 and §5 were half wrong about progress.** They said a fitting page
  should reuse [`model/progress.ts`](../../frontend/src/model/progress.ts)'s *indeterminate*
  mode plus its own iteration counter. The first half is the wrong reach: epochs, trees and
  rows are hyperparameters the user set a moment ago, so a fit's progress is
  **determinate** — it simply is not bytes. It reports `{ done, total }` over the shared
  envelope's `partial` arm, `progress.ts` is neither touched nor reused, and a shared module
  written around a byte count does not grow a second meaning for one category.
- **The fit is a `run`, not the load.** What `load` does here is hand typed arrays already
  in the tab to a worker in the same tab; it spends nothing. What costs is the fit. So the
  fit is a request, which is exactly what puts its metrics on `partial` — progress *inside*
  one run, correlated to its id — with Machine A staying `ready` and `running` staying an
  inflight count. The FIT button posts the `load` and its `run` together and the engine
  serialises the two rather than trusting message order.

[`/training`](../../frontend/src/routes/training.tsx) is the precedent for the substitution
and is the documented exception in
[`../standards/model-page-pattern.md`](../standards/model-page-pattern.md) §7: it keeps its
own full-bleed canvas rather than the four bands. **These pages do not claim that
exemption** — they have a genuine input surface and a genuine result, so they use
`ModelPage` like everything else.

Two things that belong in OUTPUT rather than in prose:

- **The confusion matrix** and **the residual plot** exist because the aggregate number
  hides the failure. In a notebook they are a chart; on a page they are the main panel.
  Both go through the lazy
  [`components/charts/EChart.tsx`](../../frontend/src/components/charts/EChart.tsx) wrapper —
  `echarts` is heavy and must stay code-split, and `npm run check:bundle` is what fails if
  it slips.
- **Feature importances.** A bar per column, and the output non-technical users actually
  read.

**The CSV parse belongs in a worker, not the component** — a ten-megabyte file parsed on the
main thread freezes the tab for a second, and the user's first interaction with the page is
a stutter. It gets [its own one-shot worker](../../frontend/src/tabular/parse.worker.ts)
rather than riding the fit worker's Machine A, and the reason is the shared envelope:
`ModelResponse`'s `ready` variant carries `{ model, backend }` and nothing else, so a parse
that rode Machine A would have no way to hand the column list back — and the page needs the
columns *before* it can offer a target.

---

## 3. Task by task

### 3.1 Tabular Classification — shipped

Route: [`/tabular-classification`](../../frontend/src/routes/tabular-classification.tsx).
Module: [`src/tabular/`](../../frontend/src/tabular/). Plan:
[#48](https://github.com/bthek1/model_playground/issues/48).

The user drops a CSV, picks a target column, and fits the ladder: **logistic regression**
as the floor, a **random forest** as the zero-tuning baseline, **gradient boosting** as the
workhorse, and an **MLP** as the deep baseline that does not win.

**Ship the MLP precisely because it loses.** That result is the point of the whole ladder,
and it is the kind of thing people do not believe until they watch it happen on their own
data. Asserting it, or omitting the MLP and explaining why, proves nothing — so it is a
genuine fit on the same split with the same metric block, and `just fe-e2e-tabular` asserts
that it comes in under the boosting model on the bundled sample, because that claim is on
screen.

Three things that had to survive the port:

- **Calibration and the threshold.** The default 0.5 is a convention, not a decision — it is
  only right when a false positive and a false negative cost the same. A slider that moves
  the threshold and updates precision, recall and the confusion matrix live is the single
  most useful control in this category, and it **re-derives on the main thread** from the
  held-out probabilities the fit already returned. It must never refit.
- **Which columns actually matter.** Permutation importance is a loop over columns with a
  shuffle in the middle, so it is a few lines and runs fast enough to be interactive. Two
  things go wrong there and both are silent: forgetting to **restore** the column poisons
  every column measured afterwards and still returns a complete, ordered, plausible ranking;
  and sharing one random stream across columns makes each column's shuffle depend on how
  many came before it, so the ranking moves when an unrelated feature is toggled off.
- **Every accuracy carries its baseline, inside the same object.** A class prior is the
  easiest thing in any dataset to learn, so a broken model and a working one both produce a
  confident, plausible number. `/graph-classification` settled this; here it travels inside
  `ClassificationMetrics` so the score and its null model can never come from different
  splits.

**The sample datasets are part of the design, not decoration.** Three ship, each with its
licence rendered beside it: **Palmer penguins** (CC0, real missing values, a categorical
column, and separable enough that the *floor* wins), **UCI wine** (CC BY 4.0, the control —
everything does well), and a **synthetic credit-risk file**. The synthetic one exists
because a sample every family gets right demonstrates nothing about any of them, and is
satisfied by a page that fits one model and draws it four times. Its risk is an
*interaction* — being leveraged is dangerous on a low income and harmless on a high one —
which no straight line can express. Measured on the shipped page at seed 42:

| sample | majority baseline | logistic | forest | boosting | MLP |
|---|---|---|---|---|---|
| credit risk (synthetic) | 0.556 | 0.611 | **0.731** | 0.700 | 0.631 |
| penguins | 0.442 | **0.988** | 0.907 | 0.965 | 0.977 |
| wine | 0.400 | **1.000** | 0.889 | **1.000** | 0.978 |

Read across the rows: the trees win where there is an interaction to find, the linear floor
wins where there is not, and the MLP never wins. That is the page.

**Encodings are fitted on the training half only** — the imputation mean, the standardiser's
variance, the level tables. Fitting them over the whole frame raises the held-out score and
looks like a better page: the `/link-prediction` leakage lesson in its cheapest form.
[`design.ts`](../../frontend/src/tabular/design.ts) takes the training indices explicitly so
the rule is impossible to forget.

### 3.2 Tabular Regression — shipped

Route: [`/tabular-regression`](../../frontend/src/routes/tabular-regression.tsx). Plan:
[#49](https://github.com/bthek1/model_playground/issues/49).

Same worker, same fit engine, same hook, same ladder — a reviewer should be able to diff the
two route files and see only the diagnostics differing. Three things genuinely differ:

- **The closed-form ridge, and the GPU/CPU boundary it makes exact.** `XᵀX` and `Xᵀy` are
  `O(n·d²)` in the row count and go to the GPU; the `d×d` factorisation is microseconds and
  stays on the CPU. §5 wants that split demonstrated rather than asserted, and this is where
  the demonstration is arithmetic rather than rhetoric.
  **And it is a Cholesky solve, not an inverse.** `XᵀX` is `d×d` with `d` in the tens, so a
  WGSL inverse would be a kernel dispatched to invert a matrix smaller than one workgroup —
  the trees lesson pointing the other way. It should not be an *inverse* either: solving
  `(XᵀX + λI)w = Xᵀy` by Cholesky is what ridge's `λ > 0` actually buys, because it makes the
  matrix positive definite and therefore the solve stable where OLS's would not be. That is
  the real reason ridge is this category's linear model rather than plain least squares, and
  the page says so.
- **Quantile regression ships an interval, and the interval is the output.** A band is more
  honest than a point estimate and reads better. Fitted by minimising the pinball loss, which
  is one line different from the squared loss and shares the gradient loop.
- **The log-transform trap is a toggle, and it spends.** Fitting on `log1p(target)` cannot be
  re-derived from a fit on the raw target, so flipping it runs nothing and the next FIT is a
  real second fit — the same class as `/video-text-to-text`'s reverse toggle. The
  demonstration is that **the error metrics are not comparable between the two**, so the page
  refuses to put an RMSE in log space beside an RMSE in the target's units without naming
  which is which; [`transform.ts`](../../frontend/src/tabular/transform.ts) owns the units so
  the route cannot mislabel them.

**Residuals replace the confusion matrix** — predicted against actual, with the residual on
the second axis. A model with a fine R² and a funnel-shaped residual plot is the case worth
seeing, and it is why the plot is a main panel rather than an extra.

**Do not inherit §3.1's hyperparameters.** A depth that suits a Gini split is not
automatically right for variance reduction, so `REGRESSION_FAMILIES` is a separate list with
its own defaults rather than an `objective` flag on the four classification entries. (The
`/link-prediction` lesson: reuse that looks like a decision is often an inheritance.)

Two findings the build added to the plan:

- **The quantile lines are fitted independently, so they can cross**, and a band whose lower
  edge is above its upper edge is not renderable. They are sorted per row, which is the
  standard repair: it changes no line's level, only which is labelled which where they have
  already crossed.
- **Coverage is the assertion, not "a band was drawn".** A band of the wrong width looks
  entirely correct on the chart. The measured fraction of held-out rows inside the 0.1–0.9
  band travels in the result and is rendered beside it, and `just fe-e2e-tabular` pins it to
  0.6–0.95 rather than to a floor.
- **Ridge's rank-deficiency report is the page's proof that the choice mattered.** At `λ = 0`
  on a design with two identical columns the Cholesky factorisation cannot complete — which is
  the honest answer, where an inverse would have returned one of infinitely many coefficient
  vectors and looked entirely fine. The page says so, adds the smallest penalty that makes the
  matrix positive definite, and finishes the fit.

### 3.3 Time Series Forecasting — shipped, as baselines and backtesting

Route: [`/time-series-forecasting`](../../frontend/src/routes/time-series-forecasting.tsx).
Module: [`src/forecast/`](../../frontend/src/forecast/). Plan:
[#50](https://github.com/bthek1/model_playground/issues/50).

This one splits.

| Component | Browser | Notes |
|---|---|---|
| The baselines you have to beat | Yes, trivially | naive, seasonal naive and drift are arithmetic. Twenty lines |
| Backtesting | Yes | a loop over rolling splits. No model needed to demonstrate the idea |
| `google/timesfm-2.0-500m-pytorch` | No | **no ONNX weights in the repo** — re-checked 2026-09-25 |
| `ibm-granite/granite-timeseries-patchtst` | No | **no ONNX weights in the repo** — re-checked 2026-09-25 |

Both are a **missing export, not a size problem**, so no amount of quantization fixes it and
it is `adding-a-task-page.md` §0 *question 1* that the row fails. A documented "no export,
here is the date it was checked" is finished work — and it does not stop the page, because
the baselines *are* the page.

What is left is honest and still worthwhile: the user pastes a series, the page draws the
baselines, and a rolling-origin backtest shows **why a single train/test split is not an
evaluation**. One split lies, and a page that lets you watch the metric swing across windows
makes that unarguable.

Four decisions worth carrying forward:

- **The backtest must be rolling-origin, not k-fold.** Shuffled cross-validation on a time
  series trains on the future and scores the past — the `/link-prediction` leakage lesson in
  its oldest form, and it fails *upward*: the metric improves. The split function is pure and
  its most valuable test asserts that **no training index is ever greater than any test index
  in the same window**.
- **One split's number is shown next to the distribution of numbers.** The whole
  demonstration is the gap between them. A page that only showed the spread would be correct
  and would not make the point.
- **MASE is the metric that makes the comparison legible**, because it is scaled by the
  in-sample naive error. It ships beside MAE and RMSE for exactly that reason — with one
  correction the build had to make: **1.0 means "no better than naive" only at a horizon of
  1.** The denominator is the in-sample *one-step* error by definition, so a multi-step
  forecast is scored against a one-step benchmark and grows away from 1 with the horizon. On
  the bundled random walk a naive forecast measures ~1.0 one step out, 1.8 at four and ~3.0 at
  fourteen, with nothing wrong. So the note beside the table is horizon-aware, and a test
  measures the ratio at three horizons and asserts it grows. The metric was right; the sentence
  explaining it was not.
- **The season length is a control, never auto-detected.** Guessing the period and being
  wrong produces a confident, plausible, wrong forecast. A control the user sets is both
  honest and the more instructive design.

**It is the only route in the repo with no worker at all**, and that is a decision rather than
an omission: there is nothing asynchronous to correlate, and wrapping pure arithmetic in a
worker to look consistent would add a protocol and a mock for no benefit. It is also a
**three-band page** — there is nothing to fit — which is written up in
[`../standards/model-page-pattern.md`](../standards/model-page-pattern.md) §7 rather than
quietly shipped, since §8's testid contract assumes four.

Three findings the build added:

- **The implied frequency is the *modal* step, not the median and not the mean.** A mean is
  dragged by one long gap far enough to make every ordinary interval look irregular — but so is
  a median on a short series: three monthly points with one month missing have two diffs,
  `[1 month, 2 months]`, whose upper median is the gap itself, and the page then reports the
  *regular* interval as the anomaly. The mode is the step that actually recurs, which is what
  "the implied frequency" means.
- **The sample had to be built so the spread is wide.** Measured on the shipped page, seasonal
  naive on the airline series gives a single-split MAE of 47.8 against a window range of
  12.6–53.1 — a 4× swing, which is far larger than the difference between the methods being
  compared. That is the finding, and a series without a regime change or a strong seasonal
  amplitude would give a tight spread and demonstrate nothing.
- **The single split is one of the windows**, by construction: the last rolling origin *is* the
  notebook split. So the E2E assertion cannot be "the spread straddles it" — that is true
  trivially. It is that the spread is **wide**, `max > 1.5 × min`, which is exactly what a
  backtest reusing one split's numbers for every window would fail.

If a foundation forecaster is genuinely wanted in the browser, the path is to export PatchTST
yourself with `torch.onnx.export` and host the artefact, then serve it through
`ModelCard.weights_url` like any other checkpoint. It is a small encoder, so it would work.
That is a project with a Python step and an artefact to host, so it gets its own plan.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | How | Best backend | If not |
|---|---|---|---|---|
| **Tabular Classification** | Yes, train and infer | TypeScript trees in a Worker; WGSL for the linear and MLP rungs | CPU for trees, WebGPU for linear | - |
| **Tabular Regression** | Yes, train and infer | as above, plus a closed-form ridge in WGSL with a CPU Cholesky | CPU for trees, WebGPU for ridge | - |
| **Time Series Forecasting**, baselines and backtesting | Yes | plain arithmetic and a rolling loop | main thread is fine | - |
| **Time Series Forecasting**, TimesFM 2.0 / PatchTST | No | no ONNX export in either repo (checked 2026-09-25) | - | server API, or export it yourself |

---

## 5. Memory and performance notes

Different constraints from every other category, because **the data is the load rather than
the model**.

- **The dataset is the memory budget.** A million-row CSV as JavaScript objects is hundreds
  of megabytes. Parse into typed arrays, one `Float32Array` per column, and never build an
  array of row objects. There is no row type anywhere in `src/tabular/`, deliberately.
- **Cap the row count, with an explicit message, and sample *evenly*.** Sampling 50 000 rows
  from a large file and saying so is better than freezing on the full file — but taking the
  first 50 000 rows of a file sorted by date is a different dataset from the one the user
  handed over, and saying "50 000 rows" about it would be true and misleading.
- **A missing value gets an explicit mask, never a sentinel.** `NaN` and `0` are both real
  values in real data, so any sentinel silently becomes one of them on some file. `Number("")`
  is `0`, which is the single most expensive coercion in JavaScript for a file parser: one
  empty cell in a numeric column shifts every mean, every split threshold and every
  coefficient that column touches, with nothing failing anywhere.
- **A ragged row is rejected with its line number, never padded.** A best-effort recovery
  produces a model fitted on shifted columns, which trains happily and is wrong everywhere.
- **Trees go in a Worker, not on the GPU**; **linear models go on the GPU.** That split is the
  honest one and demonstrating it is itself a teaching point — see §1.
- **Report progress by iteration, not by bytes** — and determinately. See §2's second
  correction.
- **There is nothing to dispose.** No GPU session outlives the fit, which makes this the one
  category where the one-model-live rule does not apply. The GPU device itself is memoised and
  shared ([`webgpu/device.ts`](../../frontend/src/webgpu/device.ts)) — do not destroy it.
- **`DeviceStatus`, not `ModelStatus`, answers the GPU question** on a page that raises one.
  There are no weights to download; the question the page actually raises is whether there is
  a GPU at all. [`/tensor`](../../frontend/src/routes/tensor.tsx) is the precedent.
- **Nothing is persisted and nothing is uploaded.** [`lib/mnistCache.ts`](../../frontend/src/lib/mnistCache.ts)
  and `lib/proteinsCache.ts` both write their dataset to IndexedDB so a reload does not
  re-download it, and both are right to — they cache a public benchmark. These pages would be
  caching someone's payroll. A reload loses the file, by design; re-dropping it costs a parse.
  **No route posts an `InferenceRun`** — `createInferenceRun` exists in `src/api/models.ts` and
  has no caller anywhere in `src/`, and these pages must not become the first: a run record
  naming the user's columns would undo the entire argument for the category. The claim is a
  behaviour, so it has an assertion.

---

## 6. Reference

- **[`src/tabular/`](../../frontend/src/tabular/)** — the columnar dataset, the CSV parser,
  the design-matrix encoder, the tree/linear/MLP/ridge/quantile fitters, the metrics and the
  fit worker.
- **[`src/forecast/`](../../frontend/src/forecast/)** — the series parser, the baselines, the
  metrics and the rolling-origin backtest. No worker, by decision.
- **[`src/webgpu/`](../../frontend/src/webgpu/)**: the existing WGSL matmul, transpose, scale
  and elementwise shaders plus `linearModel.ts` — which is reused rather than reimplemented,
  softmax, cross-entropy, SGD loop and all. [`/training`](../../frontend/src/routes/training.tsx)
  and [`/tensor`](../../frontend/src/routes/tensor.tsx) are the two routes to read first.
- **`just fe-e2e-tabular`** — the whole ladder fitted in a real browser, pinned above the
  majority baseline. **`just fe-e2e-forecast`** — the window spread against the single-split
  number, and a Worker count of zero. There is **no `fe-e2e-models` row for this category**: no checkpoint, no catalogue,
  nothing on the Hub to resolve.
- **`node scripts/make-tabular-samples.mjs`** rebuilds the bundled sample CSVs and prints
  their row counts.
- **Page construction**: [`../guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **In-repo standards**: [`../standards/model-page-pattern.md`](../standards/model-page-pattern.md)
  §7 (where the pattern bends for fitting and three-band pages),
  [`../standards/model-visualization.md`](../standards/model-visualization.md).
