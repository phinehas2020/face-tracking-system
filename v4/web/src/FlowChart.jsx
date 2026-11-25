import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const HOUR = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit" });

export default function FlowChart({ buckets }) {
  const data = buckets.map((bucket) => ({ time: HOUR.format(new Date(bucket.start)), Entries: bucket.in, Exits: bucket.out }));
  return <ResponsiveContainer width="100%" height="100%">
    <BarChart data={data} margin={{ top: 6, right: 4, bottom: 0, left: -22 }} barCategoryGap={2}>
      <CartesianGrid stroke="#d8d5ce" vertical={false} />
      <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#6e777b" }} tickLine={false} axisLine={{ stroke: "#c6c3bc" }} minTickGap={36} />
      <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "#6e777b" }} tickLine={false} axisLine={false} />
      <Tooltip cursor={{ fill: "#eeece7" }} contentStyle={{ fontSize: 12, borderRadius: 0, border: "1px solid #c6c3bc", background: "#f8f7f3" }} />
      <Bar dataKey="Entries" stackId="flow" fill="#138d86" isAnimationActive={false} />
      <Bar dataKey="Exits" stackId="flow" fill="#dc6f3b" isAnimationActive={false} />
    </BarChart>
  </ResponsiveContainer>;
}
