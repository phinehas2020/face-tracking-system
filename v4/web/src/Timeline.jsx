import { memo, useMemo } from "react";
import {
  Bar, CartesianGrid, ComposedChart, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";

const HEALTH_KEYS = ["lane_a_health", "lane_b_health", "wide_health"];
const NUMBER_FORMATTER = new Intl.NumberFormat("en-US");
const TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  hour: "numeric", minute: "2-digit", second: "2-digit",
});

function formatNumber(value) {
  return NUMBER_FORMATTER.format(value ?? 0);
}

function formatTime(value) {
  return TIME_FORMATTER.format(new Date(value));
}

function TimelineTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  return <div className="chart-tooltip"><b>{item.label}</b><span>{item.passages_per_minute} passages/min</span><span>{formatNumber(item.fusion_count)} fusion identities</span></div>;
}

const PassagesChart = memo(function PassagesChart({ chartData }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 2, right: 8, bottom: 0, left: -22 }}>
        <CartesianGrid stroke="#d9d6cf" vertical={false} />
        <XAxis dataKey="label" tick={{ fontSize: 9, fill: "#727a80" }} interval={15} axisLine={false} tickLine={false} />
        <YAxis yAxisId="left" tick={{ fontSize: 9, fill: "#727a80" }} axisLine={false} tickLine={false} />
        <Tooltip content={<TimelineTooltip />} />
        <Bar yAxisId="left" dataKey="passages_per_minute" fill="#2873c4" radius={[1, 1, 0, 0]} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
});

const EvidenceBands = memo(function EvidenceBands({ data }) {
  return (
    <>
      <div className="event-bands">
        <div className="event-row"><b>Identity merges</b><div>{data.map((item) => <i key={`merge-${item.sample_index}`} className={item.accepted_merges >= 3 ? "merge-event" : "quiet"} />)}</div></div>
        <div className="event-row"><b>Unresolved intervals</b><div>{data.map((item) => <i key={`review-${item.sample_index}`} className={item.unresolved_intervals > 0 ? "review-event" : "quiet"} />)}</div></div>
      </div>
      <div className="health-bands dense-health">
        {HEALTH_KEYS.map((key) => (
          <div className="health-row" key={key}>
            <b>{key.replace("_health", "").replace("_", " ")}</b>
            <div>{data.map((item) => <i key={`${key}-${item.sample_index}`} className={item[key] < 0.9 ? "warning" : "healthy"} />)}</div>
          </div>
        ))}
      </div>
    </>
  );
});

const DivergenceChart = memo(function DivergenceChart({ chartData }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 2, right: 8, bottom: 1, left: 2 }}>
        <YAxis domain={[-300, 150]} hide />
        <XAxis dataKey="sample_index" hide />
        <Line type="monotone" dataKey="baseline_delta" stroke="#747b7e" dot={false} strokeWidth={1} isAnimationActive={false} />
        <Line type="monotone" dataKey="fusion_delta" stroke="#236dc1" dot={false} strokeWidth={1.4} isAnimationActive={false} />
        <Line type="monotone" dataKey="candidate_delta" stroke="#c04d41" dot={false} strokeWidth={1} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
});

export default function Timeline({ data, progress, onSeek }) {
  const chartData = useMemo(() => data.map((item) => ({
    ...item,
    label: formatTime(item.timestamp).replace(/:\d{2} /, " "),
    baseline_delta: item.baseline_count - item.ground_truth_count,
    fusion_delta: item.fusion_count - item.ground_truth_count,
    candidate_delta: item.candidate_count - item.ground_truth_count,
  })), [data]);
  const progressIndex = Math.round(progress * (data.length - 1));
  return (
    <section className="timeline-card dense-timeline">
      <header className="section-heading compact">
        <div><span className="eyebrow">Event timeline</span><h2>Thu, Aug 27, 2025 · 7:00 AM–8:00 PM</h2></div>
        <div className="timeline-anchor"><b>{formatTime(data[progressIndex]?.timestamp)}</b><span>playhead {Math.round(progress * 100)}%</span></div>
      </header>
      <div className="timeline-chart passages-chart">
        <PassagesChart chartData={chartData} />
        <div className="timeline-cursor" style={{ left: `${progress * 100}%` }}><span>{formatTime(data[progressIndex]?.timestamp)}</span></div>
      </div>
      <EvidenceBands data={data} />
      <div className="divergence-row">
        <b>Model divergence<br /><small>Δ unique</small></b>
        <div className="divergence-chart">
          <DivergenceChart chartData={chartData} />
        </div>
      </div>
      <input className="timeline-scrubber" type="range" min="0" max="1000" value={Math.round(progress * 1000)} onChange={(event) => onSeek(Number(event.target.value) / 1000)} aria-label="Replay position" />
    </section>
  );
}
