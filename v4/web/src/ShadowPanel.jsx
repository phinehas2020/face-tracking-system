const NAMES = {
  buffalo_l: "buffalo_l",
  adaface_ir101_wf12m: "AdaFace IR101",
  "kprpe_wf12m-scrfd": "AdaFace ViT KP-RPE",
  "kprpe_wf4m-scrfd": "AdaFace ViT KP-RPE (4M)",
  antelopev2: "antelopev2",
};

// Other recognizers run on the same faces in their own process. Nothing here feeds the count.
export function ShadowPanel({ shadow }) {
  const report = shadow?.report;
  if (!shadow?.enabled && !report) return null;
  const running = report && Date.now() - new Date(report.updated_at).getTime() < 90000;
  const rated = Boolean(report?.rated);
  const cell = (value) => !value?.n ? "—" : rated && value["tar@far0.01"] != null
    ? `${Math.round(value["tar@far0.01"] * 100)}% matched · ${value.n} pairs`
    : `${value.median.toFixed(2)} similarity · ${value.n} pairs`;
  const models = report ? Object.entries(report.models ?? {}) : [];
  const turned = report ? report.scored_faces - (report.accepted_by_live_gate ?? 0) : 0;
  return <section className="console-panel console-shadow">
    <h2><strong>Recognizer comparison</strong><span>Shadow run · does not affect counts or visitors</span><em className={running ? "ok" : ""}>{running ? "Running" : "Not running"}</em></h2>
    {!report ? <p className="console-empty">Faces are being saved for comparison. The comparison process has not reported yet.</p> : <>
      <p className="console-shadow-note">{report.faces} face crops from {report.tracks} followed faces · {turned} turned further than the live recognizer accepts</p>
      {models.length ? <table>
        <thead><tr><th>Recognizer</th><th>Same person, facing</th><th>Same person, turned</th><th>Different people</th><th>People (estimate)</th><th>Speed</th></tr></thead>
        <tbody>{models.map(([name, row]) => <tr key={name}>
          <th>{NAMES[name] ?? name}<span>{name === report.live_model ? "live" : "shadow"}</span></th>
          <td>{cell(row.summary?.frontal)}</td>
          <td>{cell(row.summary?.turned)}</td>
          <td>{row.impostor?.n ? `${row.impostor.p99.toFixed(2)} at worst 1%` : "—"}</td>
          <td>{row.people ? `${row.people.all_angles.people} · ${row.people.frontal_only.people} facing only` : "—"}</td>
          <td>{row.ms_per_face != null ? `${Math.round(row.ms_per_face)} ms` : name === "buffalo_l" ? "with detection" : "—"}</td>
        </tr>)}</tbody>
      </table> : <p className="console-empty">No face has been seen since the comparison started.</p>}
      {models.length && !rated ? <p className="console-shadow-note">Match rates appear after {report.min_people_pairs} different pairs of people have been seen together in one picture; {report.people_pairs} so far. Until then the similarity of two crops of one followed face is shown: higher is better.</p> : null}
    </>}
  </section>;
}
