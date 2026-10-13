import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Forecast a time series with naive, seasonal naive and drift baselines, and backtest them over rolling windows.",
  about: [
    "Time series forecasting predicts the next values of a sequence from its past: monthly passengers, daily sales, a price. This page draws the baselines any forecasting model has to beat, namely the naive forecast (repeat the last value), seasonal naive (repeat the last season), drift (extend the average change per step) and the historical mean. Paste a column of numbers or date,value pairs, drop a CSV, or start from the airline passengers series or one of two synthetic samples, then set the horizon and the season length.",
    "The second half of the page is backtesting. A rolling-origin backtest recomputes the forecast at a series of cut-off points, with an expanding or sliding training window, and scores each forecast only on data after its cut-off, so the future never leaks into training the way shuffled k-fold cross-validation would let it. The error from the usual single train/test split is shown next to the spread across windows: on the airline series, seasonal naive scores an MAE of 47.8 on the single split while individual windows range from 12.6 to 53.1. Errors are reported as MAE, RMSE, MAPE and MASE, where MASE scales by the in-sample one-step naive error.",
    "There is no learned model here. The foundation forecasters TimesFM and PatchTST publish no ONNX weights, so they cannot run in a browser, and these baselines are plain arithmetic on the main thread. Nothing is downloaded and your series is not uploaded. Gaps and irregular spacing in the dates are reported rather than filled in.",
  ],
  aliases: [
    "time series forecasting",
    "forecasting baselines",
    "seasonal naive",
    "backtesting",
    "rolling-origin backtest",
    "MASE",
  ],
};
