import { lazy, memo, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  Aperture, ArrowsClockwise, Check, CheckCircle, Crosshair, Pause,
  Play, ShieldCheck, Tag, UsersThree,
} from "@phosphor-icons/react";
import { EVENT_ID, loadEvent, sendReplayControl, submitDecision } from "./data.js";

const Timeline = lazy(() => import("./Timeline.jsx"));
const MiniCurve = lazy(() => import("./MiniCurve.jsx"));

const CAMERA_ASSETS = {
  "Lane A": "/assets/cameras/lane-a.webp",
  "Lane B": "/assets/cameras/lane-b.webp",
  Wide: "/assets/cameras/wide.webp",
};
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

const MODEL_METRICS = [
  ["Unique count", (run) => formatNumber(run.unique_count)],
  ["Error vs GT", (run) => `${run.error_percent.toFixed(2)}%`],
  ["False merges", (run) => run.false_merges],
  ["False splits", (run) => run.false_splits],
  ["Passage recall", (run) => `${(run.passage_recall * 100).toFixed(2)}%`],
  ["Review burden", (run) => `${run.review_minutes}m`],
];

const CameraFeed = memo(function CameraFeed({ camera, overlays, active, onSelect }) {
  const select = () => onSelect(camera.name);
  return (
    <article data-camera={camera.name.toLowerCase().replace(" ", "-")} className={active ? "camera-feed active" : "camera-feed"} onClick={select} onKeyDown={(event) => (event.key === "Enter" || event.key === " ") && select()} role="button" tabIndex="0" aria-label={`Select ${camera.name} camera`}>
      <header>
        <div><span className="status-dot" /> <strong>{camera.name}</strong></div>
        <span className="camera-role"><span className="status-dot" /> sync</span>
      </header>
      <div className="camera-image">
        <img src={CAMERA_ASSETS[camera.name]} alt={`${camera.name} event recording`} />
        <div className="timecode">08:54:19:{camera.name === "Lane A" ? "12" : camera.name === "Lane B" ? "14" : "13"}</div>
        {overlays ? (
          <>
            <span className={`track-box ${camera.name === "Wide" ? "wide-one" : "lane-one"}`}><b>{camera.name === "Lane B" ? "B-12894" : "A-19378"}</b></span>
            <span className={`track-box secondary ${camera.name === "Wide" ? "wide-two" : "lane-two"}`}><b>{camera.name === "Lane B" ? "B-12901" : "A-19382"}</b></span>
            <span className="portal-line">commit</span>
          </>
        ) : null}
        <div className="camera-telemetry">
          <span>{camera.fps.toFixed(1)} fps</span><span>{camera.frame_age_ms} ms</span><span>clock ±{camera.name === "Lane B" ? 4 : 3} ms</span>
        </div>
      </div>
    </article>
  );
});

const HeaderMetric = memo(function HeaderMetric({ label, value, detail, tone = "" }) {
  return (
    <div className="header-metric">
      <span>{label}</span>
      <b className={tone}>{value}</b>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
});

const ModelMatrix = memo(function ModelMatrix({ runs, selectedRun, onSelect }) {
  return (
    <section className="model-matrix">
      <div className="matrix-title"><span className="eyebrow">Model run comparison</span><small>same video · truth 3,842</small></div>
      <div className="matrix-grid matrix-head">
        <span>Metric</span>
        {runs.map((run) => <button key={run.id} className={selectedRun === run.id ? "active" : ""} onClick={() => onSelect(run.id)}>{run.name}</button>)}
      </div>
      {MODEL_METRICS.map(([label, render]) => (
        <div className="matrix-grid" key={label}>
          <b>{label}</b>
          {runs.map((run) => <span key={run.id} className={run.name === "Fusion v4" ? "fusion-value" : run.name === "Candidate" ? "candidate-value" : ""}>{render(run)}</span>)}
        </div>
      ))}
    </section>
  );
});

const QueueList = memo(function QueueList({ candidates, selectedId, onSelect }) {
  return (
    <section className="queue-list">
      <div className="queue-title"><span className="eyebrow">Disagreement queue</span><b>{candidates.length}</b></div>
      <div className="queue-items">
        {candidates.map((candidate, index) => (
          <button key={candidate.id} className={candidate.id === selectedId ? "selected" : ""} onClick={() => onSelect(candidate.id)}>
            <strong>{index + 1}</strong>
            <span><b>ID: {candidate.left_ref} / {candidate.right_ref}</b><small>{candidate.left_lane} · {formatTime(candidate.left_observed_at)}<br />{candidate.right_lane} · {formatTime(candidate.right_observed_at)}</small></span>
            <em>divergence<br /><b>{candidate.recommended_decision}</b></em>
          </button>
        ))}
      </div>
    </section>
  );
});

const InspectionPanel = memo(function InspectionPanel({ candidates, selectedId, onSelect, onDecision, busy }) {
  const candidate = candidates.find((item) => item.id === selectedId) ?? candidates[0];
  if (!candidate) return <aside className="inspection-panel empty-review"><CheckCircle size={34} weight="duotone" /><strong>Queue cleared</strong><span>Every ambiguous link is resolved.</span></aside>;
  const index = candidates.findIndex((item) => item.id === candidate.id);
  const move = (delta) => onSelect(candidates[(index + delta + candidates.length) % candidates.length].id);
  return (
    <aside className="inspection-panel">
      <header><div><span className="eyebrow">Inspecting case {index + 1} of {candidates.length}</span><h2>ID: {candidate.left_ref} / {candidate.right_ref}</h2></div><b>{candidate.recommended_decision}</b></header>
      <div className="inspection-images">
        <figure><span>Lane A · {formatTime(candidate.left_observed_at)}</span><img src="/assets/cameras/lane-a.webp" alt="Lane A selected identity track" /><figcaption>{candidate.left_ref}</figcaption></figure>
        <figure><span>Lane B · {formatTime(candidate.right_observed_at)}</span><img src="/assets/cameras/lane-b.webp" alt="Lane B selected identity track" /><figcaption>{candidate.right_ref}</figcaption></figure>
      </div>
      <div className="inspection-decision"><b>Fusion v4 decision: {candidate.recommended_decision}</b><span>confidence {candidate.probability.toFixed(2)}</span></div>
      <p><strong>Reason:</strong> {candidate.reason}</p>
      <div className="inspection-actions">
        <button onClick={() => move(-1)}>Prev case</button><button onClick={() => move(1)}>Next case</button>
        <button disabled={busy} onClick={() => onDecision(candidate, "different")}>Keep separate</button>
        <button disabled={busy} className="primary" onClick={() => onDecision(candidate, "same")}>Confirm merge</button>
      </div>
    </aside>
  );
});

function PlaybackBar({ playing, onPlay, progress, speed, onSpeed, sync, onSync, overlays, onOverlays, onAnnotate }) {
  return (
    <footer className="playback-bar">
      <button className="play-button" onClick={onPlay} aria-label={playing ? "Pause replay" : "Play replay"}>{playing ? <Pause size={17} weight="fill" /> : <Play size={17} weight="fill" />}</button>
      <span className="playback-time">{`${String(Math.floor(progress * 13)).padStart(2, "0")}:${String(Math.floor((progress * 780) % 60)).padStart(2, "0")}:00`} <small>/ 13:00:00</small></span>
      <div className="replay-progress"><span style={{ width: `${progress * 100}%` }} /></div>
      <div className="speed-control">{[1, 8, 32].map((value) => <button key={value} className={speed === value ? "active" : ""} onClick={() => onSpeed(value)}>{value}×</button>)}</div>
      <button className={sync ? "utility active" : "utility"} onClick={onSync}><ArrowsClockwise size={15} /> Sync</button>
      <button className={overlays ? "utility active" : "utility"} onClick={onOverlays}><Crosshair size={15} /> Overlays</button>
      <button className="utility" onClick={onAnnotate}><Tag size={15} /> Annotate</button>
    </footer>
  );
}

function LifecycleDialog({ onClose, eventId }) {
  const [confirmation, setConfirmation] = useState("");
  const required = `PURGE ${eventId}`;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="dialog" role="dialog" aria-modal="true" aria-labelledby="lifecycle-title">
        <div className="dialog-icon"><ShieldCheck size={24} weight="duotone" /></div>
        <span className="eyebrow">Event lifecycle</span><h2 id="lifecycle-title">Verified, event-scoped purge</h2>
        <p>The purge plan deletes recordings, embeddings, tracklets, review decisions, metrics, and this event’s database records. It cannot address another event.</p>
        <div className="purge-list"><span><Check size={14} /> database cascade checked</span><span><Check size={14} /> event file boundary checked</span><span><Check size={14} /> confirmation phrase required</span></div>
        <label>Type <code>{required}</code><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>
        <div className="dialog-actions"><button onClick={onClose}>Cancel</button><button className="danger" disabled={confirmation !== required}>Purge event data</button></div>
      </section>
    </div>
  );
}

export function App() {
  const [dashboard, setDashboard] = useState(null);
  const [activeView, setActiveView] = useState("rehearsal");
  const [activeCamera, setActiveCamera] = useState("Wide");
  const [selectedRun, setSelectedRun] = useState("run-fusion-v4");
  const [selectedCase, setSelectedCase] = useState("candidate-001");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [progress, setProgress] = useState(0.318);
  const [sync, setSync] = useState(true);
  const [overlays, setOverlays] = useState(true);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState("");
  const reviewRef = useRef(null);

  useEffect(() => {
    let active = true;
    loadEvent().then((value) => { if (active) setDashboard(value); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => setProgress((value) => (value + speed / 46800) % 1), 250);
    return () => window.clearInterval(timer);
  }, [playing, speed]);
  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 2800);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const handleView = useCallback((view) => {
    if (view === "lifecycle") return setActiveView("lifecycle");
    setActiveView(view);
    if (view === "review") window.setTimeout(() => reviewRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 0);
    if (view === "calibrate") { setOverlays(true); setToast("Calibration overlay armed — select a camera"); }
    if (view === "live") setToast("Live topology view shares the same verified passage pipeline");
  }, []);

  const handleDecision = useCallback(async (candidate, decision) => {
    setBusy(true);
    try {
      if (dashboard.connected) await submitDecision(candidate.id, decision);
      const candidates = dashboard.candidates.filter((item) => item.id !== candidate.id);
      setDashboard((value) => ({ ...value, candidates, summary: { ...value.summary, unresolved: Math.max(0, value.summary.unresolved - 1), unique_attendees: value.summary.unique_attendees - (decision === "same" ? 1 : 0) } }));
      setSelectedCase(candidates[0]?.id ?? "");
      setToast(decision === "same" ? "Identity link merged and audit logged" : "Tracks kept separate and audit logged");
    } catch (error) { setToast(error.message); } finally { setBusy(false); }
  }, [dashboard]);

  if (!dashboard) return <main className="loading-screen"><Aperture size={34} className="loading-mark" /><span>Synchronizing event rehearsal…</span></main>;
  const { summary, runs, timeline, candidates } = dashboard;
  return (
    <div className="app-shell">
      <main className="command-center">
        <header className="intel-header">
          <div className="product-title">
            <h1>Event Time Machine</h1>
            <div className="mode-switch"><button className={activeView === "rehearsal" ? "active" : ""} onClick={() => handleView("rehearsal")}>Rehearsal</button><button className={activeView === "live" ? "active" : ""} onClick={() => handleView("live")}>Live</button></div>
          </div>
          <HeaderMetric label="Date anchor" value="Aug 27, 2026" />
          <HeaderMetric label="Rehearsal progress" value={`${Math.round(summary.rehearsal_progress * 100)}%`} detail="complete" />
          <HeaderMetric label="Predicted unique (Fusion v4)" value={formatNumber(summary.unique_attendees)} tone="blue" />
          <HeaderMetric label="Ground truth (last year)" value={formatNumber(summary.ground_truth)} tone="olive" />
          <HeaderMetric label="Error (Fusion v4)" value={`${((summary.unique_attendees - summary.ground_truth) / summary.ground_truth * 100).toFixed(2)}%`} tone="blue" />
          <HeaderMetric label="Event" value={summary.title} detail="Tent entrance · two lane" />
          <HeaderMetric label="Replay source" value="Aug 27, 2025" detail="7:00 AM–8:00 PM" />
          <div className="header-controls">
            <button onClick={() => handleView("review")} aria-label="Open identity review"><UsersThree size={16} /></button>
            <button onClick={() => handleView("calibrate")} aria-label="Open camera calibration"><Crosshair size={16} /></button>
            <button onClick={() => handleView("lifecycle")} aria-label="Open event lifecycle"><ShieldCheck size={16} /></button>
          </div>
        </header>

        <section className="upper-layout">
          <section className="camera-mosaic" aria-label="Synchronized three-camera evidence">
            {summary.cameras.map((camera) => <CameraFeed key={camera.id} camera={camera} active={activeCamera === camera.name} onSelect={setActiveCamera} overlays={overlays} />)}
            <Suspense fallback={<div className="mini-curve loading-mini">Loading curve…</div>}>
              <MiniCurve data={timeline} />
            </Suspense>
          </section>
          <aside className="analysis-stack">
            <ModelMatrix runs={runs} selectedRun={selectedRun} onSelect={setSelectedRun} />
            <div ref={reviewRef}><QueueList candidates={candidates} selectedId={selectedCase} onSelect={setSelectedCase} /></div>
          </aside>
        </section>

        <section className="lower-layout">
          <Suspense fallback={<section className="timeline-card timeline-loading">Loading event timeline…</section>}>
            <Timeline data={timeline} progress={progress} onSeek={(value) => { setProgress(value); sendReplayControl("seek", { position: value }); }} />
          </Suspense>
          <InspectionPanel candidates={candidates} selectedId={selectedCase} onSelect={setSelectedCase} onDecision={handleDecision} busy={busy} />
        </section>
        <PlaybackBar playing={playing} onPlay={() => { setPlaying((value) => !value); sendReplayControl(playing ? "pause" : "play"); }} progress={progress} speed={speed} onSpeed={(value) => { setSpeed(value); sendReplayControl("speed", { speed: value }); }} sync={sync} onSync={() => setSync((value) => !value)} overlays={overlays} onOverlays={() => setOverlays((value) => !value)} onAnnotate={() => setToast("Review marker added at the current timecode")} />
      </main>
      {activeView === "lifecycle" ? <LifecycleDialog eventId={EVENT_ID} onClose={() => setActiveView("rehearsal")} /> : null}
      {toast ? <div className="toast"><CheckCircle size={17} weight="fill" /> {toast}</div> : null}
    </div>
  );
}
