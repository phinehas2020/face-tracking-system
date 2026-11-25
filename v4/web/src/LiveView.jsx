import { useEffect, useRef, useState } from "react";
import { Aperture, Crosshair, Pause, Play } from "@phosphor-icons/react";
import { LineForm } from "./LineForm.jsx";
import { LiveConsole } from "./LiveConsole.jsx";
import "./live.css";

async function api(path, options) {
  const response = await fetch(`/api/live/${path}`, { cache: "no-store", ...options });
  if (!response.ok) throw new Error(`Camera service returned ${response.status}`);
  return response.json();
}

export function LiveView() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [calibration, setCalibration] = useState(null);
  const [busy, setBusy] = useState(false);
  const [drawingGate, setDrawingGate] = useState(false);
  const [firstGatePoint, setFirstGatePoint] = useState(null);
  const [frame, setFrame] = useState(null);
  const [facingFrame, setFacingFrame] = useState(null);
  const [frameError, setFrameError] = useState(false);
  const objectUrl = useRef(null);
  const [history, setHistory] = useState(null);

  useEffect(() => {
    document.title = "Ingress · Live camera test";
    let stopped = false;
    let timer;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const next = await api("status", { signal: controller.signal });
        if (stopped) return;
        setStatus(next);
        setCalibration((current) => current ?? next.calibration);
        setError("");
      } catch (failure) {
        if (!stopped) setError(failure.message);
      }
      if (!stopped) timer = setTimeout(poll, 750);
    };
    poll();
    return () => { stopped = true; clearTimeout(timer); controller.abort(); };
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const response = await fetch("/api/live/frame.jpg", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("No fresh frame");
        const blob = await response.blob();
        if (stopped) return;
        const url = URL.createObjectURL(blob);
        const old = objectUrl.current;
        objectUrl.current = url;
        setFrame(url);
        setFrameError(false);
        if (old) URL.revokeObjectURL(old);
      } catch {
        if (!stopped) { setFrameError(true); setFrame(null); }
      }
      if (!stopped) timer = setTimeout(poll, 150);
    };
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller.abort();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  useEffect(() => {
    if (!status?.is_dual_live) return;
    let stopped = false, timer, previous;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const response = await fetch("/api/live/frame.jpg?camera=facing", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error("No facing frame");
        const blob = await response.blob();
        if (stopped) return;
        const next = URL.createObjectURL(blob);
        if (previous) URL.revokeObjectURL(previous);
        previous = next;
        setFacingFrame(next);
      } catch { if (!stopped) setFacingFrame(null); }
      if (!stopped) timer = setTimeout(poll, 200);
    };
    poll();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); if (previous) URL.revokeObjectURL(previous); };
  }, [status?.is_dual_live]);

  useEffect(() => {
    if (!status?.is_dual_live) return;
    let stopped = false, timer;
    const poll = async () => {
      // A station running older software has no history; the chart is then left out.
      try { const next = await api("history?hours=6&bucket_minutes=10"); if (!stopped) setHistory(next); }
      catch { if (!stopped) setHistory(null); }
      if (!stopped) timer = setTimeout(poll, 20000);
    };
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [status?.is_dual_live]);

  const mutate = async (path, method, data) => {
    setBusy(true);
    try {
      await api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
      setStatus(await api("status"));
      setError("");
    } catch (failure) { setError(failure.message); }
    finally { setBusy(false); }
  };
  const dual = Boolean(status?.is_dual_live);
  const replay = Boolean(status?.is_replay);
  const paired = Boolean(status?.is_paired);
  const complete = status?.status === "complete";
  const connected = !error && !frameError && (!dual || (facingFrame && status?.station_status === "live")) && ["live", "paused", "complete"].includes(status?.status);
  const entryDirection = calibration?.gate_start ? "along the orange arrow" : calibration?.axis === "vertical"
    ? (calibration.entry_positive ? "left → right" : "right → left")
    : (calibration?.entry_positive ? "top → bottom" : "bottom → top");
  const dirty = calibration && ["axis", "position", "entry_positive", "require_body_motion", "anchor", "counting_mode", "segment_start", "segment_end", "gate_start", "gate_end"].some(
    (key) => JSON.stringify(calibration[key]) !== JSON.stringify(status?.calibration?.[key]),
  );

  const chooseGatePoint = (event) => {
    if (!drawingGate || !calibration) return;
    const box = event.currentTarget.getBoundingClientRect();
    const point = [Math.max(0, Math.min(1, (event.clientX-box.left)/box.width)), Math.max(0, Math.min(1, (event.clientY-box.top)/box.height))];
    if (!firstGatePoint) setFirstGatePoint(point);
    else {
      setCalibration({ ...calibration, counting_mode: "trajectory", gate_start: firstGatePoint, gate_end: point });
      setFirstGatePoint(null); setDrawingGate(false);
    }
  };
  const runtimeLabel = (runtime) => `${runtime?.detector_device === "mps" ? "Apple GPU" : runtime?.detector_device || "Loading"} detection · ${runtime?.face_provider === "CoreMLExecutionProvider" ? "CoreML" : runtime?.face_provider === "CPUExecutionProvider" ? "CPU" : "Loading"} face analysis`;

  if (dual) return <LiveConsole status={status} error={error} frame={frame} facingFrame={facingFrame} connected={connected} history={history} calibration={calibration} setCalibration={setCalibration} mutate={mutate} busy={busy} dirty={dirty} drawingGate={drawingGate} setDrawingGate={setDrawingGate} firstGatePoint={firstGatePoint} setFirstGatePoint={setFirstGatePoint} chooseGatePoint={chooseGatePoint} entryDirection={entryDirection} />;

  return <main className="live-shell">
    <header className="live-header">
      <div className="live-brand"><Aperture size={30} /><div><span className="eyebrow">Ingress · {replay ? "Recorded-camera rehearsal" : "Facial recognition test"}</span><h1>{dual ? "Facing recognition + side crossings" : paired ? "Station 2 · Front + side rehearsal" : replay ? "2025 Fair · Crowded sections" : "Recognize visitors. Confirm entry."}</h1></div></div>
      <div className={`live-status ${connected ? "connected" : ""}`} role="status">
        <span className="status-dot" />{error ? "Service unavailable" : complete ? "Replay complete" : connected ? (status.paused ? "Paused" : replay ? "Playing recording" : "Camera live") : status?.status === "error" ? "Processing error" : replay ? "Loading recording" : "Connecting camera"}
      </div>
    </header>
    {error || status?.error ? <p className="live-alert" role="alert">{error || status.error}</p> : null}
    {status?.enabled === false ? <p className="live-alert">No live camera is configured for this server.</p> : null}
    {replay ? <section className="live-replay-source"><div><strong>{status.source?.label || "Loading selected excerpts"}</strong><p>{status.source?.file}</p><small>{paired ? status.synchronization_note : `Clip ${status.source?.clip ?? 1} of ${status.source?.total_clips ?? "—"} · ${status.source?.playback_speed ?? 1}× maximum video speed · Every frame processed · Camera clocks are not synchronized`}</small></div><div><progress aria-label="Excerpt progress" max={status.source?.duration_seconds || 1} value={Math.max(0,(status.source?.position_seconds || 0)-(status.source?.start_seconds || 0))} /><span>{Math.max(0,Math.round((status.source?.position_seconds || 0)-(status.source?.start_seconds || 0)))} / {Math.round(status.source?.duration_seconds || 0)} seconds</span>{complete ? <button className="live-primary" disabled={busy} onClick={() => mutate("replay/start","POST",{})}>Start a fresh replay</button> : null}</div></section> : null}
    {paired ? <section className="paired-summary" aria-label="Paired-camera comparison">
      <div><strong>Two views · one entry counter</strong><p>Independent camera recognition, combined only through accepted evidence. Only the side view counts entries.</p></div>
      <div><b>{status.comparison?.side_only_linked_entries ?? 0}</b><span>Entries linked with side camera alone</span></div>
      <div><b>{status.cross_view_identity_merges?.length ?? 0}</b><span>Independently enrolled profiles consolidated across views</span></div>
      <div><b>{status.comparison?.paired_linked_entries ?? 0}</b><span>Entries linked in paired run · net change {status.comparison?.net_link_gain ?? 0}</span></div>
    </section> : null}
    {paired ? <p className="live-alert">{status.appearance_fusion_enabled ? "Repeated person-appearance matching is active; uncertain links remain unresolved. " : "Direct face sharing is active. "}These results are not verified attendance totals.{status.cached_observations ? " Replay uses cached model observations from these source frames." : ""}</p> : null}
    <section className={`live-metrics ${dual ? "dual-metrics" : ""}`} aria-label="Live counts">
      {dual ? <><div><span>Entries</span><b>{status?.counts?.in ?? "—"}</b><small>Confirmed physical crossings into the room</small></div><div><span>Exits</span><b>{status?.counts?.out ?? "—"}</b><small>Confirmed physical crossings out of the room</small></div></> : null}
      <div><span>Face-linked entrants</span><b>{status?.recognition?.unique_attendees ?? "—"}</b><small>Distinct face profiles linked to an entry</small></div>
      <div><span>Repeat entries</span><b>{status?.recognition?.repeat_entries ?? "—"}</b><small>Returning visitors, counted once above</small></div>
      <div><span>Unresolved entries</span><b>{status?.recognition?.unresolved_entries ?? "—"}</b><small>Crossed the line; face not yet matched</small></div>
      <div><span>Face profiles</span><b>{status?.recognition?.known_faces ?? "—"}</b><small>Model-enrolled profiles; not a verified headcount</small></div>
    </section>
    <div className="live-recognition-status" role="status"><strong>{status?.face_status === "ready" ? (complete ? "Recognition complete" : "Face recognition active") : status?.face_status === "error" ? "Face recognition error" : "Loading face recognition"}</strong><span>{connected && !status?.paused ? status.people_visible : "—"} people in {complete ? "last frame" : "view"}</span><span>{status?.counts?.in ?? 0} entries · {status?.counts?.out ?? 0} exits</span><span>{status?.face_samples_processed ?? 0} face samples analyzed</span></div>
    <div className={`live-grid ${paired ? "paired-grid" : ""}`}>
      <section className="live-camera">
        <header><strong>{replay ? status.source?.label : dual ? "Side · passage counting" : "Camera 01"}<span>{replay ? "Recorded footage" : `ANNKE ${status?.model || ""}`}</span></strong><span>{replay ? status.source?.station : status?.camera || "Local camera"}</span></header>
        <div className={`live-image ${drawingGate ? "drawing-gate" : ""}`} onClick={chooseGatePoint}>
          {frame && !error ? <img src={frame} alt={replay ? "Recorded fair footage with face recognition and passage tracking" : "Live ANNKE camera with detected people and crossing line"} /> : <div className="live-wait"><Aperture size={42} /><strong>Waiting for a fresh frame</strong><span>{replay ? "Loading the selected recording excerpt." : "The live image appears here when the camera connects."}</span></div>}
          {drawingGate ? <span className="live-pause-label">{firstGatePoint ? "Click the second end of the doorway gate" : "Click the first end of the doorway gate"}</span> : null}
          {status?.paused ? <span className="live-pause-label">COUNTING PAUSED</span> : null}
        </div>
        <footer><span>{status?.resolution?.join(" × ") || "—"}</span><span>{complete ? "Last processed frame" : connected ? `${status.processing_fps} ${status.cached_observations ? "cached replay" : "processing"} fps` : "Feed unavailable"}</span><span>{complete ? "Replay ended" : status?.frame_age_ms != null ? `${status.frame_age_ms} ms frame age` : "—"}</span><span>{status?.frames_processed?.toLocaleString() ?? 0} frames processed</span></footer>
        {dual ? <div className="live-runtime">{runtimeLabel(status?.runtime)}</div> : null}
        <div className="live-face-tracks">{status?.active_faces?.length ? status.active_faces.map((face) => <div key={face.track}><strong>{face.label}</strong><span>{face.state === "new" ? "New face" : face.state === "matched" ? "Known face matched" : face.state === "tracked" ? "Identity retained by continuous tracking" : face.state === "conflict" ? "Conflicting evidence — unlinked" : face.state === "review" ? "Unresolved — needs review" : `Waiting for ${face.required_samples || 3} agreeing samples`}</span>{face.similarity != null ? <small>Similarity {face.similarity.toFixed(2)}</small> : null}</div>) : <p>{paired ? "Front and side are independently tracked. Visitor labels show accepted face evidence; a body box alone is not an identity match." : "Looking for a clear face. Face recognition assigns a visitor label after several consistent views."}</p>}</div>
      </section>
      {dual ? <section className="live-camera facing-camera">
        <header><strong>Facing · face recognition<span>ANNKE {status.facing?.model}</span></strong><span>{status.facing?.camera}</span></header>
        <div className="live-image">{facingFrame ? <img src={facingFrame} alt="Facing camera recognizing visitors, without a counting line" /> : <div className="live-wait"><Aperture size={42} /><strong>Reconnecting facing camera</strong></div>}</div>
        <footer><span>{status.facing?.resolution?.join(" × ")}</span><span>{status.facing?.status}</span><span>{status.facing?.frame_age_ms ?? "—"} ms frame age</span><span>{status.facing?.frames_processed ?? 0} frames processed</span></footer>
        <div className="live-runtime">{runtimeLabel(status?.facing?.runtime)}</div>
        <div className="live-face-tracks"><p>Recognizes faces and supplies appearance evidence for matching people between views. Crossings are counted only in the side view. A visitor without matching face evidence stays unresolved.</p>{status.facing?.active_faces?.map(face => <div key={face.track}><strong>{face.label}</strong><span>{face.state}</span></div>)}</div>
      </section> : null}
      <aside className="live-settings">
        <div className="live-section-title"><Crosshair size={20} /><h2>{dual ? "Side camera · confirm entry" : "Confirm entry"}</h2></div>
        <p>{dual ? "The facing view identifies faces; only the side view counts crossings. " : ""}Faces identify visitors. The orange line confirms that they entered. Position it across the walking path.{paired ? " Paired calibration changes apply to the next replay." : ""}</p>
        <LineForm calibration={calibration} setCalibration={setCalibration} mutate={mutate} busy={busy} dirty={dirty} replay={replay} paired={paired} complete={complete} drawingGate={drawingGate} setDrawingGate={setDrawingGate} setFirstGatePoint={setFirstGatePoint} entryDirection={entryDirection} />
        <button className="live-pause" disabled={busy || !status?.enabled || complete} onClick={() => mutate("control", "POST", { paused: !status.paused })}>{status?.paused ? <Play size={17} /> : <Pause size={17} />}{status?.paused ? "Resume" : replay ? "Pause replay" : "Pause counting"}</button>
        <div className="live-test-note"><strong>{replay ? "Recorded-camera test" : "Recognition test"}</strong><p>{replay ? (paired ? "Both views advance on a reconstructed common clock. Each camera recognizes faces independently. Repeated appearance and timestamp agreement support cross-view links; validated local matches are preserved. Precise geometric projection remains disabled because it failed validation." : "These crowded excerpts use the same face recognition and passage tracking as the live camera. Each fresh replay has its own counts. Results are test measurements, not verified attendance totals.") : "Face the camera until a visitor label appears. Enter across the line, leave, then enter again. The same visitor should remain one unique entrant, with one repeat entry."}</p><small>Face templates and counts stay on this Mac for this test event. Visitor thumbnails are saved locally with this event and removed when the event is purged. Live video stays in memory; explicit validation recordings are saved locally. Matching thresholds are provisional; uncertain faces remain unresolved.</small></div>
      </aside>
    </div>
    {replay && !paired ? <section className="live-history"><h2>Station results <span>This replay run only</span></h2><table><thead><tr><th>Station / view</th><th>Passages</th><th>Entries</th><th>Unique entrants</th><th>Unresolved entries</th></tr></thead><tbody>{status.station_results?.map((row) => <tr key={row.camera}><td>{row.camera}</td><td>{row.passages}</td><td>{row.entries}</td><td>{row.unique_entrants}</td><td>{row.unresolved_entries}</td></tr>)}</tbody></table><p>Peak people tracked in a frame: {status.peak_people_visible ?? 0}. Usable face detections: {status.face_detections ?? 0}; face observations: {status.face_samples_processed ?? 0}; body-linked samples: {status.body_face_samples ?? 0}. Processed {status.frames_processed ?? 0} of {status.frames_received ?? status.frames_processed ?? 0} received frames; {status.track_handovers ?? 0} track handovers preserved.</p></section> : null}
    {dual ? <section className="live-history"><h2>Camera matching</h2><p>{status.cross_camera_association?.linked ?? 0} crossings linked between cameras since service start · {status.cross_camera_association?.pending ?? 0} awaiting identity evidence. Body appearance must agree across multiple frames; uncertain crossings stay counted.</p><p>Physical walk-through validation is still required after changing camera placement or the gate.</p></section> : null}
    <section className="live-history"><h2>Recognized visitors <span>Stored face identities for this test</span></h2>
      <div className="live-visitors">{status?.recognition?.visitors?.map((visitor) => <article key={visitor.identity_id}>{visitor.has_preview ? <img src={`/api/live/faces/${visitor.identity_id}/preview.jpg`} alt={`${visitor.label} saved face preview`} /> : <div className="live-face-empty">Photo pending</div>}<div><strong>{visitor.label}</strong><p>{visitor.entries ? `${visitor.entries} ${visitor.entries === 1 ? "entry" : "entries"} · ${visitor.entries - 1} repeat` : "Face recognized · no entry yet"}</p><small>{visitor.samples} enrollment samples</small></div></article>)}</div>
      {!status?.recognition?.visitors?.length ? <p>No face enrolled yet. Look toward the camera for a moment.</p> : null}
    </section>
    <section className="live-history"><h2>Recent passages <span>Face identity + crossing evidence</span></h2>
      {status?.recent?.length ? <table><thead><tr><th>{replay ? "Source position" : "Time"}</th><th>Direction</th><th>Visitor</th><th>Identity evidence</th></tr></thead><tbody>{status.recent.map((item) => <tr key={`${item.observed_at}-${item.track_id}`}><td>{replay && item.recording ? `${item.recording.station} · ${Math.floor(item.recording.position_seconds / 60)}:${String(Math.floor(item.recording.position_seconds % 60)).padStart(2,"0")}` : new Date(item.observed_at).toLocaleTimeString()}</td><td>{item.direction === "in" ? "Entry" : "Exit"}</td><td>{item.label || "Unresolved"}</td><td>{(item.reviewed_replay || item.recording?.validation === "real_time_live_replay") ? <small className="reviewed-recording">Reviewed recording · </small> : null}{item.status === "face_matched" ? (item.identity_basis === "cross_view_appearance" ? "Face linked from other camera" : item.identity_basis === "continuous_track" ? "Retained from prior face" : item.identity_basis === "later_face_match" ? "Face linked after crossing" : "Face matched") : "No usable face match"}</td></tr>)}</tbody></table> : <p>No crossings recorded yet. Walk across the orange line to begin.</p>}
    </section>
  </main>;
}
