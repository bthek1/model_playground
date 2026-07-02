import { createFileRoute } from "@tanstack/react-router"
import { lazy, Suspense } from "react"
import type { EChartsOption } from "echarts"
import {
  Bar,
  CartesianGrid,
  Legend,
  Line,
  Tooltip,
  ComposedChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts"

const EChart = lazy(() => import("@/components/charts/EChart"))

export const Route = createFileRoute("/demo/chart")({
  component: ChartDemoPage,
})

const categories = [1, 2, 3, 4, 5]
const seriesA = [2, 6, 3, 8, 5]
const seriesB = [4, 3, 7, 1, 6]

const echartsOption: EChartsOption = {
  grid: { top: 40, right: 20, bottom: 40, left: 50 },
  tooltip: { trigger: "axis" },
  legend: {},
  xAxis: { type: "category", data: categories.map(String) },
  yAxis: { type: "value" },
  series: [
    { name: "Series A", type: "line", smooth: true, data: seriesA },
    { name: "Series B", type: "bar", data: seriesB },
  ],
}

const rechartsData = categories.map((x, i) => ({
  x,
  "Series A": seriesA[i],
  "Series B": seriesB[i],
}))

function ChartDemoPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-10 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Chart Demo</h1>
        <p className="text-sm text-muted-foreground">
          ECharts (lazy-loaded) and Recharts examples.
        </p>
      </div>

      <section>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          ECharts
        </h2>
        <Suspense
          fallback={
            <div className="text-muted-foreground">Loading chart…</div>
          }
        >
          <EChart
            option={echartsOption}
            className="h-96 w-full rounded-lg border"
          />
        </Suspense>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          Recharts
        </h2>
        <div className="h-96 w-full rounded-lg border p-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rechartsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Series B" fill="var(--chart-2)" />
              <Line
                type="monotone"
                dataKey="Series A"
                stroke="var(--chart-1)"
                strokeWidth={2}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  )
}
