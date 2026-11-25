import { ArrowRight } from "@phosphor-icons/react";

// The counting-line controls, shared by every live and replay layout.
export function LineForm({ calibration, setCalibration, mutate, busy, dirty, replay, paired, complete, drawingGate, setDrawingGate, setFirstGatePoint, entryDirection }) {
  if (!calibration) return null;
  return <form onSubmit={(event) => { event.preventDefault(); mutate("calibration", "PUT", calibration); }}>
    {!replay ? <>
      <label>Counting method<select value={calibration.counting_mode || "legacy"} onChange={event => setCalibration({ ...calibration, counting_mode: event.target.value })}><option value="trajectory">Confirmed doorway traversal</option><option value="legacy">Legacy frame counter</option></select></label>
      <label>Tracked position<select value={calibration.anchor || "footpoint"} onChange={event => setCalibration({ ...calibration, anchor: event.target.value })}><option value="center">Visible body center</option><option value="footpoint">Bottom of body box</option></select></label>
      <button type="button" className="live-pause" onClick={() => { setDrawingGate(!drawingGate); setFirstGatePoint(null); }}>{drawingGate ? "Cancel gate drawing" : "Draw gate on side camera"}</button>
    </> : null}
    <label>Line orientation<select value={calibration.axis} onChange={(event) => setCalibration({ ...calibration, axis: event.target.value, gate_start: null, gate_end: null })}><option value="vertical">Vertical line</option><option value="horizontal">Horizontal line</option></select></label>
    <label>Line position <strong>{Math.round(calibration.position * 100)}%</strong><input type="range" min="5" max="95" step="1" value={Math.round(calibration.position * 100)} onChange={(event) => setCalibration({ ...calibration, position: Number(event.target.value) / 100, gate_start: null, gate_end: null })} /></label>
    <label>Entry direction<select value={String(calibration.entry_positive)} onChange={(event) => setCalibration({ ...calibration, entry_positive: event.target.value === "true" })}><option value="true">{calibration.gate_start ? "Forward across drawn gate" : calibration.axis === "vertical" ? "Left to right" : "Top to bottom"}</option><option value="false">{calibration.gate_start ? "Reverse across drawn gate" : calibration.axis === "vertical" ? "Right to left" : "Bottom to top"}</option></select></label>
    <label className="live-motion-option"><input type="checkbox" checked={Boolean(calibration.require_body_motion)} onChange={(event) => setCalibration({ ...calibration, require_body_motion: event.target.checked })} /> Require body movement across line</label><div className="live-direction"><ArrowRight size={18} /> Entry: {entryDirection}</div>
    <button className="live-primary" disabled={busy || !dirty || (paired && !complete)}>{busy ? "Saving…" : dirty ? "Apply line" : "Line saved"}</button>
  </form>;
}
