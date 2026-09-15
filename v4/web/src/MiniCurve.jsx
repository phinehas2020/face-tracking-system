import { memo } from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";

export default memo(function MiniCurve({ data }) {
  const compact = data.filter((_, index) => index % 3 === 0);
  return (
    <article className="mini-curve">
      <header><span>Unique count curve comparison</span><b>Ground truth · 3,842</b></header>
      <div className="mini-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={compact} margin={{ top: 7, right: 8, bottom: 0, left: -29 }}>
            <XAxis dataKey="sample_index" hide />
            <YAxis domain={[0, 4000]} tick={{ fontSize: 7, fill: "#788084" }} axisLine={false} tickLine={false} />
            <Line type="monotone" dataKey="ground_truth_count" stroke="#7b7a3d" dot={false} strokeWidth={1.2} />
            <Line type="monotone" dataKey="fusion_count" stroke="#1f6ec2" dot={false} strokeWidth={1.7} />
            <Line type="monotone" dataKey="baseline_count" stroke="#71787b" dot={false} strokeWidth={1.1} />
            <Line type="monotone" dataKey="candidate_count" stroke="#bd4437" dot={false} strokeWidth={1.1} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <footer><span>7 AM</span><span>1 PM</span><span>8 PM</span></footer>
    </article>
  );
});
