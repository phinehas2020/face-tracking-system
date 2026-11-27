import { lazy, Suspense, useEffect, useState } from "react";
import { Aperture, CaretDown, CaretUp, Crosshair, FilmStrip, Pause, Play, SignIn, SignOut } from "@phosphor-icons/react";
import { LineForm } from "./LineForm.jsx";
import { ShadowPanel } from "./ShadowPanel.jsx";
import "./console.css";

const FlowChart = lazy(() => import("./FlowChart.jsx"));
const CLOCK = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });
const DAY = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" });

const provider = (runtime) => runtime?.face_provider === "CoreMLExecutionProvider" ? "CoreML faces" : runtime?.face_provider === "CPUExecutionProvider" ? "CPU faces" : "faces loading";
const device = (runtime) => runtime?.detector_device === "mps" ? "Apple GPU" : runtime?.detector_device || "loading";
const evidence = (item) => item.status === "face_matched"
  ? (item.identity_basis === "cross_view_appearance" ? "Linked from other camera" : item.identity_basis === "continuous_track" ? "Retained from prior face" : item.identity_basis === "later_face_match" ? "Linked after crossing" : "Face matched")
  : "No usable face";

function Camera({ title, cam, frame, alt, waiting, drawing, onClick, children }) {
  const [width, height] = cam?.resolution?.length === 2 ? cam.resolution : [4, 3];
  const live = cam?.status === "live" && frame;
  return <section className="console-camera">
    <header><strong>{title}</strong><span>{cam?.model} · {cam?.camera}</span><em className={live ? "ok" : ""}>{live ? "Live" : cam?.status || "Connecting"}</em></header>
    {/* The well takes the camera's own shape, so a click maps straight onto the picture. */}
    <div className={`console-well ${drawing ? "drawing" : ""}`} style={{ aspectRatio: `${width} / ${height}` }} onClick={onClick}>
      {frame ? <img src={frame} alt={alt} /> : <div className="console-wait"><Aperture size={36} /><strong>{waiting}</strong></div>}
      {children}
    </div>
    <footer><span>{cam?.processing_fps ?? "—"} fps</span><span>{cam?.frame_age_ms ?? "—"} ms behind</span><span>{cam?.people_visible ?? 0} in view</span><span>{device(cam?.runtime)} · {provider(cam?.runtime)}</span></footer>
  </section>;
}

export function LiveConsole({ status, error, frame, facingFrame, connected, history, shadow, calibration, setCalibration, mutate, busy, dirty, drawingGate, setDrawingGate, firstGatePoint, setFirstGatePoint, chooseGatePoint, entryDirection }) {
  const [now, setNow] = useState(() => new Date());
  const [lineOpen, setLineOpen] = useState(false);
  useEffect(() => { const timer = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(timer); }, []);

  const facing = status?.facing;
  const recognition = status?.recognition;
  const visitors = recognition?.visitors ?? [];
  const recent = status?.recent ?? [];
  const buckets = history?.buckets ?? null;
  const lastHour = buckets ? buckets.slice(-Math.round(3600 / history.bucket_seconds)).reduce((sum, bucket) => ({ in: sum.in + bucket.in, out: sum.out + bucket.out }), { in: 0, out: 0 }) : null;
  const pose = facing?.runtime?.face_pose_gate;
  const posed = pose ? pose.accepted + pose.rejected_angle : 0;
  const link = status?.cross_camera_association;
  const formOpen = lineOpen || dirty || drawingGate;
  const lineSummary = !calibration ? "Loading" : calibration.gate_start ? "Drawn gate" : `${calibration.axis === "vertical" ? "Vertical" : "Horizontal"} line at ${Math.round(calibration.position * 100)}%`;
  const state = error ? "Service unavailable" : connected ? (status.paused ? "Counting paused" : "Station live") : status?.status === "error" ? "Processing error" : "Connecting cameras";

  const side = status?.side_entries;
  const tiles = [
    ...(side ? [["Side entries", side.count, `${side.last_hour} in the last hour · left to right, tent floor`]] : []),
    ["Entries", status?.counts?.in, lastHour ? `${lastHour.in} in the last hour` : "Counted crossings in"],
    ["Exits", status?.counts?.out, lastHour ? `${lastHour.out} in the last hour` : "Counted crossings out"],
    ["In view now", connected && !status?.paused ? status.people_visible : null, "People on the counting camera"],
    ["Face-linked", recognition?.unique_attendees, "Distinct faces tied to an entry"],
    ["Repeat entries", recognition?.repeat_entries, "Returning visitors"],
    ["Unresolved", recognition?.unresolved_entries, "Entries with no face match"],
    ["Face profiles", recognition?.known_faces, "Enrolled; not a headcount"],
  ];

  return <main className="console">
    <header className="console-top">
      <div className="console-brand"><Aperture size={28} /><div><span className="eyebrow">Ingress · Live station</span><h1>{status?.event_id || "Station"}</h1></div></div>
      <div className="console-top-right">
        <span className="console-clock">{DAY.format(now)} · {CLOCK.format(now)}</span>
        <a className="console-link" href="/api/live/review"><FilmStrip size={16} />Review clips</a>
        <div className={`console-state ${connected && !status?.paused ? "ok" : ""}`} role="status"><span className="status-dot" />{state}</div>
      </div>
    </header>
    {error || status?.error || facing?.error ? <p className="live-alert" role="alert">{error || status?.error || facing?.error}</p> : null}

    <section className="console-tiles" aria-label="Live counts" style={{ "--tiles": tiles.length }}>
      {tiles.map(([label, value, note]) => <div key={label}><span>{label}</span><b>{value ?? "—"}</b><small>{note}</small></div>)}
    </section>

    <div className="console-body">
      <div className="console-main">
        <div className="console-cameras">
          <Camera title="Side · counting" cam={status} frame={error ? null : frame} alt="Side camera with tracked people and the counting line" waiting="Waiting for the side camera" drawing={drawingGate} onClick={chooseGatePoint}>
            {firstGatePoint ? <span className="console-gate-point" style={{ left: `${firstGatePoint[0] * 100}%`, top: `${firstGatePoint[1] * 100}%` }} /> : null}
            {drawingGate ? <span className="console-well-label">{firstGatePoint ? "Click the other end of the gate" : "Click one end of the gate"}</span> : null}
            {status?.paused ? <span className="console-well-label">COUNTING PAUSED</span> : null}
          </Camera>
          <Camera title="Facing · faces" cam={facing} frame={facingFrame} alt="Facing camera recognizing visitors" waiting="Reconnecting the facing camera" />
        </div>
        {buckets ? <section className="console-panel console-flow">
          <h2><strong>Passages over time</strong><span>Last 6 hours · 10-minute bars</span><i className="key in" />Entries<i className="key out" />Exits</h2>
          <div className="console-chart"><Suspense fallback={null}><FlowChart buckets={buckets} /></Suspense></div>
        </section> : null}
        <section className="console-panel console-health">
          <h2>Station health</h2>
          <dl>
            <div><dt>Counting speed</dt><dd>{status?.processing_fps ?? "—"} fps{status?.runtime?.fps_with_people != null ? ` · ${status.runtime.fps_with_people} with people` : ""}</dd></div>
            <div><dt>Face camera speed</dt><dd>{facing?.processing_fps ?? "—"} fps{facing?.runtime?.fps_with_people != null ? ` · ${facing.runtime.fps_with_people} with people` : ""}</dd></div>
            <div><dt>Faces usable</dt><dd>{posed ? `${pose.accepted} of ${posed} · ${Math.round(100 * pose.accepted / posed)}%` : "None seen yet"}</dd></div>
            <div><dt>Camera matching</dt><dd>{link ? `${link.linked} linked · ${link.pending} pending` : "—"}</dd></div>
            <div><dt>Detector</dt><dd>{status?.runtime?.detector_model || "—"} at {status?.runtime?.detector_imgsz || "—"} px</dd></div>
            <div><dt>Camera restarts</dt><dd>{(status?.capture_restarts ?? 0) + (facing?.capture_restarts ?? 0)}</dd></div>
          </dl>
        </section>
        <ShadowPanel shadow={shadow} />
      </div>

      <aside className="console-side">
        <section className={`console-panel console-line ${formOpen ? "open" : ""}`}>
          <h2><Crosshair size={18} />Counting line{dirty ? <em>Not applied yet</em> : null}</h2>
          <p className="console-line-summary">{lineSummary} · entry {entryDirection}</p>
          <div className="console-line-actions">
            <button type="button" onClick={() => setLineOpen(!formOpen)} disabled={dirty || drawingGate}>{formOpen ? <CaretUp size={14} /> : <CaretDown size={14} />}{formOpen ? "Hide settings" : "Adjust line"}</button>
            <button type="button" disabled={busy || !status?.enabled} onClick={() => mutate("control", "POST", { paused: !status.paused })}>{status?.paused ? <Play size={14} /> : <Pause size={14} />}{status?.paused ? "Resume" : "Pause"}</button>
          </div>
          {formOpen ? <LineForm calibration={calibration} setCalibration={setCalibration} mutate={mutate} busy={busy} dirty={dirty} replay={false} paired={false} complete={false} drawingGate={drawingGate} setDrawingGate={setDrawingGate} setFirstGatePoint={setFirstGatePoint} entryDirection={entryDirection} /> : null}
        </section>

        <section className="console-panel">
          <h2>Recognized visitors <span>{visitors.length}</span></h2>
          {visitors.length ? <div className="console-visitors">{visitors.map((visitor) => <article key={visitor.identity_id}>
            {visitor.has_preview ? <img src={`/api/live/faces/${visitor.identity_id}/preview.jpg`} alt={`${visitor.label} saved face`} /> : <div className="console-face-empty">Pending</div>}
            <div><strong>{visitor.label}</strong><p>{visitor.entries ? `${visitor.entries} ${visitor.entries === 1 ? "entry" : "entries"}${visitor.entries > 1 ? ` · ${visitor.entries - 1} repeat` : ""}` : "Seen · no entry yet"}</p></div>
            <small>{visitor.samples} samples</small>
          </article>)}</div> : <p className="console-empty">No face enrolled yet.</p>}
        </section>

        <section className="console-panel">
          <h2>Recent passages <span>Latest {recent.length}</span></h2>
          {recent.length ? <ul className="console-passages">{recent.map((item) => <li key={`${item.observed_at}-${item.track_id}`}>
            <time>{new Date(item.observed_at).toLocaleTimeString()}</time>
            <b className={item.direction === "in" ? "in" : "out"}>{item.direction === "in" ? <SignIn size={14} /> : <SignOut size={14} />}{item.direction === "in" ? "Entry" : "Exit"}</b>
            <span>{item.label || evidence(item)}</span>
          </li>)}</ul> : <p className="console-empty">No crossings yet. Walk across the orange line.</p>}
        </section>
      </aside>
    </div>
  </main>;
}
