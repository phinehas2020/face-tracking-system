export const EVENT_ID = "community-harvest-2026-rehearsal";

const cameras = [
  { id: "cam-lane-a", name: "Lane A", lane: "A", status: "healthy", fps: 29.97, frame_age_ms: 42 },
  { id: "cam-lane-b", name: "Lane B", lane: "B", status: "healthy", fps: 29.97, frame_age_ms: 37 },
  { id: "cam-wide", name: "Wide", lane: null, status: "healthy", fps: 29.97, frame_age_ms: 41 },
];

export const fallbackSummary = {
  id: EVENT_ID,
  title: "Community Harvest Festival",
  event_date: "2026-08-27",
  source_label: "Aug 27, 2025 · 7:00 AM–8:00 PM",
  raw_passages: 4417,
  unique_attendees: 3841,
  repeat_passages: 576,
  unresolved: 3,
  rehearsal_progress: 0.72,
  ground_truth: 3842,
  cameras,
};

export const fallbackRuns = [
  { id: "run-baseline", name: "Baseline", model_bundle: "YOLOv8 + frame matching", unique_count: 3612, error_percent: -6, false_merges: 124, false_splits: 186, passage_recall: 0.9731, review_minutes: 312 },
  { id: "run-fusion-v4", name: "Fusion v4", model_bundle: "YOLO26 + track templates + fusion", unique_count: 3841, error_percent: -0.03, false_merges: 12, false_splits: 28, passage_recall: 0.9923, review_minutes: 41 },
  { id: "run-candidate", name: "Candidate", model_bundle: "YOLO26 + greedy nearest neighbor", unique_count: 3702, error_percent: -3.64, false_merges: 47, false_splits: 83, passage_recall: 0.9798, review_minutes: 128 },
];

export const fallbackCandidates = [
  { id: "candidate-001", left_ref: "A-19378", right_ref: "B-12894", left_lane: "A", right_lane: "B", left_observed_at: "2025-08-27T08:54:18.112-05:00", right_observed_at: "2025-08-27T08:54:20.945-05:00", face_similarity: 0.94, body_similarity: 0.81, face_quality: 0.91, body_quality: 0.87, supporting_frames: 12, probability: 0.92, status: "review", recommended_decision: "merge", reason: "High appearance similarity, consistent direction, temporal proximity, and wide-camera trajectory overlap.", evidence: { time_delta_seconds: 2.833, wide_overlap: true, candidate_margin: 0.18 } },
  { id: "candidate-002", left_ref: "A-17521", right_ref: "B-11207", left_lane: "A", right_lane: "B", left_observed_at: "2025-08-27T10:31:02.334-05:00", right_observed_at: "2025-08-27T10:31:03.119-05:00", face_similarity: 0.71, body_similarity: 0.88, face_quality: 0.63, body_quality: 0.92, supporting_frames: 8, probability: 0.64, status: "review", recommended_decision: "split", reason: "Body evidence is strong, but face geometry and simultaneous wide-camera tracks conflict.", evidence: { time_delta_seconds: 0.785, wide_overlap: false, candidate_margin: 0.04 } },
  { id: "candidate-003", left_ref: "A-21011", right_ref: "B-13655", left_lane: "A", right_lane: "B", left_observed_at: "2025-08-27T14:18:44.887-05:00", right_observed_at: "2025-08-27T14:18:47.221-05:00", face_similarity: 0.86, body_similarity: 0.76, face_quality: 0.82, body_quality: 0.79, supporting_frames: 9, probability: 0.84, status: "review", recommended_decision: "merge", reason: "Face templates agree across pose; timing and lane topology support one attendee.", evidence: { time_delta_seconds: 2.334, wide_overlap: true, candidate_margin: 0.11 } },
];

export const fallbackTimeline = Array.from({ length: 96 }, (_, index) => {
  const passages = Math.max(5, Math.round(27 + 18 * Math.sin(index / 7.5) + 11 * Math.sin(index / 2.6) + 76 * Math.exp(-(((index - 50) / 18) ** 2))));
  const truth = Math.round((3842 * (index + 1)) / 96);
  return {
    sample_index: index,
    timestamp: new Date(Date.UTC(2025, 7, 27, 12, index * 8)).toISOString(),
    passages_per_minute: passages,
    accepted_merges: (index * 7) % 6,
    unresolved_intervals: [32, 33, 62, 63, 78].includes(index) ? 2 : index % 27 === 0 ? 1 : 0,
    lane_a_health: [63, 64].includes(index) ? 0.72 : 1,
    lane_b_health: index === 33 ? 0.84 : 1,
    wide_health: [78, 79].includes(index) ? 0.79 : 1,
    baseline_count: Math.round(truth * 0.94),
    fusion_count: Math.round(truth * 0.9997),
    candidate_count: Math.round(truth * 0.9636),
    ground_truth_count: truth,
  };
});

async function getJson(path) {
  const response = await fetch(path, { signal: AbortSignal.timeout(1800) });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

export async function loadEvent() {
  try {
    const [summary, runs, timeline, candidates] = await Promise.all([
      getJson(`/api/events/${EVENT_ID}/summary`),
      getJson(`/api/events/${EVENT_ID}/replay-runs`),
      getJson(`/api/events/${EVENT_ID}/timeline`),
      getJson(`/api/events/${EVENT_ID}/candidates?status=review`),
    ]);
    return { summary, runs: runs.runs, timeline: timeline.samples, candidates: candidates.candidates, connected: true };
  } catch {
    return { summary: fallbackSummary, runs: fallbackRuns, timeline: fallbackTimeline, candidates: fallbackCandidates, connected: false };
  }
}

export async function submitDecision(candidateId, decision) {
  const response = await fetch(`/api/events/${EVENT_ID}/candidates/${candidateId}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, actor: "rehearsal-operator" }),
  });
  if (!response.ok) throw new Error("Decision could not be saved");
  return response.json();
}

export async function sendReplayControl(action, payload = {}) {
  try {
    await fetch(`/api/events/${EVENT_ID}/replay/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...payload }),
    });
  } catch {
    // The local UI remains fully usable with its deterministic offline fixture.
  }
}
