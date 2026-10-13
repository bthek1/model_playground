import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Fit ridge, tree and quantile regression models to your own CSV and predict a numeric column.",
  about: [
    "Tabular regression predicts a continuous value, such as a price, a weight or a ratio, from the other columns of a table. Load a CSV or a sample, choose the numeric target, and fit ridge regression, a random forest, gradient boosting or quantile regression on a held-out split. The result is scored with RMSE and R² in the target's own units, and a residual plot of predicted against actual sits beside the numbers, because a model with a respectable R² can still be systematically wrong at one end of the range.",
    "Ridge regression is solved in closed form. The heavy part, XᵀX and Xᵀy over every row, runs as WGSL on your GPU through WebGPU, and the small d×d system is then solved by a Cholesky factorisation on the CPU rather than inverted. The penalty is what makes that factorisation legitimate: with no penalty and two identical columns it cannot complete, and the page says so instead of returning one of infinitely many answers. The trees run in TypeScript in a Web Worker.",
    "Quantile regression returns a prediction interval rather than a point, a band from the 0.1 to the 0.9 quantile, with its measured coverage of the held-out rows shown beside it, since a band of the wrong width looks just as convincing on a chart. A log transform toggle fits on log1p of the target, and the page keeps its errors in the original units so the two fits can be compared fairly. Your file is not uploaded or stored anywhere; it stays in the tab and a reload clears it.",
  ],
  aliases: [
    "tabular regression",
    "regression model",
    "ridge regression",
    "quantile regression",
    "prediction interval",
    "residual plot",
    "log transform",
  ],
};
