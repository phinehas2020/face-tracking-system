from __future__ import annotations

import json
import math
import random
from datetime import UTC, datetime, timedelta

from .domain import utc_now
from .store import EventStore

DEMO_EVENT_ID = "community-harvest-2026-rehearsal"


def seed_demo(store: EventStore) -> str:
    if store.event_exists(DEMO_EVENT_ID):
        return DEMO_EVENT_ID

    rng = random.Random(260827)
    now = utc_now()
    with store.transaction() as connection:
        connection.execute(
            """
            INSERT INTO events
                (id, title, event_date, status, source_label, started_at, ended_at,
                 retention_until, created_at)
            VALUES (?, ?, ?, 'live', ?, ?, ?, ?, ?)
            """,
            (
                DEMO_EVENT_ID,
                "Community Harvest Festival",
                "2026-08-27",
                "Aug 27, 2025 · 7:00 AM–8:00 PM",
                "2026-08-27T07:00:00-05:00",
                "2026-08-27T20:00:00-05:00",
                "2026-08-28T23:59:59-05:00",
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO event_counters
                (event_id, raw_passages, unique_attendees, repeat_passages,
                 unresolved, rehearsal_progress, ground_truth, updated_at)
            VALUES (?, 4417, 3841, 576, 3, 0.72, 3842, ?)
            """,
            (DEMO_EVENT_ID, now),
        )

        cameras = [
            (
                "cam-lane-a",
                "Lane A",
                "entry_face",
                "A",
                "demo://lane-a",
                29.97,
                42,
                {"approach_zone": [[0.08, 0.1], [0.92, 0.1], [0.82, 0.72], [0.18, 0.72]],
                 "commit_line": [[0.18, 0.72], [0.82, 0.72]]},
            ),
            (
                "cam-lane-b",
                "Lane B",
                "entry_face",
                "B",
                "demo://lane-b",
                29.97,
                37,
                {"approach_zone": [[0.08, 0.1], [0.92, 0.1], [0.82, 0.72], [0.18, 0.72]],
                 "commit_line": [[0.18, 0.72], [0.82, 0.72]]},
            ),
            (
                "cam-wide",
                "Wide",
                "shared_overview",
                None,
                "demo://wide",
                29.97,
                41,
                {"lane_a_polygon": [[0.02, 0.12], [0.49, 0.12], [0.46, 0.95], [0.02, 0.95]],
                 "lane_b_polygon": [[0.51, 0.12], [0.98, 0.12], [0.98, 0.95], [0.54, 0.95]]},
            ),
        ]
        for camera_id, name, role, lane, source, fps, frame_age, config in cameras:
            connection.execute(
                """
                INSERT INTO cameras
                    (id, event_id, name, role, lane, source, status, fps,
                     frame_age_ms, configuration)
                VALUES (?, ?, ?, ?, ?, ?, 'healthy', ?, ?, ?)
                """,
                (
                    camera_id,
                    DEMO_EVENT_ID,
                    name,
                    role,
                    lane,
                    source,
                    fps,
                    frame_age,
                    json.dumps(config),
                ),
            )

        runs = [
            (
                "run-baseline", "Baseline", "YOLOv8 + frame face matching",
                3612, -6.00, 124, 186, 0.9731, 312,
            ),
            (
                "run-fusion-v4", "Fusion v4", "YOLO26 + track templates + fusion",
                3841, -0.03, 12, 28, 0.9923, 41,
            ),
            (
                "run-candidate", "Candidate", "YOLO26 + greedy nearest neighbor",
                3702, -3.64, 47, 83, 0.9798, 128,
            ),
        ]
        for run_id, name, bundle, count, error, merges, splits, recall, review in runs:
            connection.execute(
                """
                INSERT INTO replay_runs
                    (id, event_id, name, model_bundle, progress, unique_count,
                     error_percent, false_merges, false_splits, passage_recall,
                     review_minutes, status, created_at)
                VALUES (?, ?, ?, ?, 0.72, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    DEMO_EVENT_ID,
                    name,
                    bundle,
                    count,
                    error,
                    merges,
                    splits,
                    recall,
                    review,
                    now,
                ),
            )

        start = datetime(2025, 8, 27, 7, 0, tzinfo=UTC)
        baseline = fusion = candidate = truth = 0
        for index in range(96):
            wave = 26 + 18 * math.sin(index / 7.5) + 11 * math.sin(index / 2.6)
            peak = 75 * math.exp(-((index - 50) / 18) ** 2)
            passages = max(4, int(wave + peak + rng.randint(-8, 11)))
            truth += max(1, int(passages * rng.uniform(0.82, 0.93)))
            baseline += max(0, int(passages * rng.uniform(0.68, 0.87)))
            fusion += max(1, int(passages * rng.uniform(0.82, 0.94)))
            candidate += max(0, int(passages * rng.uniform(0.74, 0.9)))
            connection.execute(
                """
                INSERT INTO timeline_samples
                    (event_id, sample_index, timestamp, passages_per_minute,
                     accepted_merges, unresolved_intervals, lane_a_health,
                     lane_b_health, wide_health, baseline_count, fusion_count,
                     candidate_count, ground_truth_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DEMO_EVENT_ID,
                    index,
                    (start + timedelta(minutes=index * 8)).isoformat(),
                    passages,
                    rng.randint(0, 6),
                    rng.randint(0, 3),
                    1.0 if index not in {63, 64} else 0.72,
                    1.0 if index not in {33} else 0.84,
                    1.0 if index not in {78, 79} else 0.79,
                    baseline,
                    fusion,
                    candidate,
                    truth,
                ),
            )

        candidates = [
            (
                "candidate-001",
                "A-19378",
                "B-12894",
                "A",
                "B",
                "2025-08-27T08:54:18.112-05:00",
                "2025-08-27T08:54:20.945-05:00",
                0.94,
                0.81,
                0.91,
                0.87,
                12,
                0.92,
                "merge",
                "High appearance similarity, consistent direction, temporal proximity, "
                "and wide-camera trajectory overlap.",
                {"time_delta_seconds": 2.833, "wide_overlap": True, "candidate_margin": 0.18},
            ),
            (
                "candidate-002",
                "A-17521",
                "B-11207",
                "A",
                "B",
                "2025-08-27T10:31:02.334-05:00",
                "2025-08-27T10:31:03.119-05:00",
                0.71,
                0.88,
                0.63,
                0.92,
                8,
                0.64,
                "split",
                "Body evidence is strong, but face geometry and simultaneous "
                "wide-camera tracks conflict.",
                {"time_delta_seconds": 0.785, "wide_overlap": False, "candidate_margin": 0.04},
            ),
            (
                "candidate-003",
                "A-21011",
                "B-13655",
                "A",
                "B",
                "2025-08-27T02:18:44.887-05:00",
                "2025-08-27T02:18:47.221-05:00",
                0.86,
                0.76,
                0.82,
                0.79,
                9,
                0.84,
                "merge",
                "Face templates agree across pose; timing and lane topology support one attendee.",
                {"time_delta_seconds": 2.334, "wide_overlap": True, "candidate_margin": 0.11},
            ),
        ]
        for candidate in candidates:
            (
                candidate_id,
                left_ref,
                right_ref,
                left_lane,
                right_lane,
                left_time,
                right_time,
                face,
                body,
                face_quality,
                body_quality,
                frames,
                probability,
                recommendation,
                reason,
                evidence,
            ) = candidate
            connection.execute(
                """
                INSERT INTO identity_links
                    (id, event_id, left_ref, right_ref, left_lane, right_lane,
                     left_observed_at, right_observed_at, face_similarity,
                     body_similarity, face_quality, body_quality, supporting_frames,
                     probability, status, recommended_decision, reason, evidence,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'review', ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    DEMO_EVENT_ID,
                    left_ref,
                    right_ref,
                    left_lane,
                    right_lane,
                    left_time,
                    right_time,
                    face,
                    body,
                    face_quality,
                    body_quality,
                    frames,
                    probability,
                    recommendation,
                    reason,
                    json.dumps(evidence),
                    now,
                ),
            )
    return DEMO_EVENT_ID
