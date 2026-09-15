PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    event_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('setup', 'live', 'closed', 'purged')),
    source_label TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    retention_until TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_counters (
    event_id TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    raw_passages INTEGER NOT NULL DEFAULT 0,
    unique_attendees INTEGER NOT NULL DEFAULT 0,
    repeat_passages INTEGER NOT NULL DEFAULT 0,
    unresolved INTEGER NOT NULL DEFAULT 0,
    rehearsal_progress REAL NOT NULL DEFAULT 0,
    ground_truth INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    lane TEXT,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    fps REAL NOT NULL DEFAULT 0,
    frame_age_ms INTEGER NOT NULL DEFAULT 0,
    clock_offset_ms INTEGER NOT NULL DEFAULT 0,
    clock_drift_ppm REAL NOT NULL DEFAULT 0,
    configuration TEXT NOT NULL DEFAULT '{}',
    UNIQUE(event_id, name)
);

CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    prototype_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS passages (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    lane TEXT NOT NULL,
    track_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    direction TEXT NOT NULL,
    identity_id TEXT REFERENCES identities(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    face_quality REAL NOT NULL DEFAULT 0,
    body_quality REAL NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '{}',
    UNIQUE(event_id, camera_id, track_id, observed_at)
);

CREATE TABLE IF NOT EXISTS identity_links (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    left_passage_id TEXT,
    right_passage_id TEXT,
    left_ref TEXT NOT NULL,
    right_ref TEXT NOT NULL,
    left_lane TEXT NOT NULL,
    right_lane TEXT NOT NULL,
    left_observed_at TEXT NOT NULL,
    right_observed_at TEXT NOT NULL,
    face_similarity REAL,
    body_similarity REAL,
    face_quality REAL NOT NULL,
    body_quality REAL NOT NULL,
    supporting_frames INTEGER NOT NULL,
    probability REAL NOT NULL,
    status TEXT NOT NULL,
    recommended_decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '{}',
    reviewed_by TEXT,
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(left_passage_id) REFERENCES passages(id) ON DELETE SET NULL,
    FOREIGN KEY(right_passage_id) REFERENCES passages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS replay_runs (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    model_bundle TEXT NOT NULL,
    progress REAL NOT NULL,
    unique_count INTEGER NOT NULL,
    error_percent REAL NOT NULL,
    false_merges INTEGER NOT NULL,
    false_splits INTEGER NOT NULL,
    passage_recall REAL NOT NULL,
    review_minutes INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeline_samples (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    passages_per_minute INTEGER NOT NULL,
    accepted_merges INTEGER NOT NULL,
    unresolved_intervals INTEGER NOT NULL,
    lane_a_health REAL NOT NULL,
    lane_b_health REAL NOT NULL,
    wide_health REAL NOT NULL,
    baseline_count INTEGER NOT NULL,
    fusion_count INTEGER NOT NULL,
    candidate_count INTEGER NOT NULL,
    ground_truth_count INTEGER NOT NULL,
    PRIMARY KEY(event_id, sample_index)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    before_state TEXT,
    after_state TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_passages_event_time ON passages(event_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_passages_identity ON passages(identity_id);
CREATE INDEX IF NOT EXISTS idx_links_event_status ON identity_links(event_id, status);
CREATE INDEX IF NOT EXISTS idx_timeline_event_index ON timeline_samples(event_id, sample_index);
