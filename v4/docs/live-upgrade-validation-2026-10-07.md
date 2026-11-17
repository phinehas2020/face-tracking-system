# Edith camera upgrade and validation

October 7, 2026

The live service now runs the upgraded two-camera pipeline on Edith. The facing camera supplies face recognition and body appearance; only the side camera counts passages. A recorded walkthrough was replayed through the actual asynchronous live pipeline with an isolated database, rather than only testing model predictions.

## Running configuration

- Apple M4, 16 GiB memory.
- YOLO26n at 960-pixel input on MPS, with a doorway-specific TrackTrack configuration.
- InsightFace buffalo_l on CoreML, including native-resolution head crops. Its face-match threshold remains 0.60; uncertain identities are not forced into a match.
- OSNet body features support same-view track recovery and matching across cameras.
- Facing stream: 3840×2160 at 15 fps. Side stream: 1920×1080 at 20 fps.
- Body-center gate at 55% of the side image, spanning 40–98% vertically, with time-based traversal confirmation. The facing view has no counting line.
- Latest-frame decode workers, bounded asynchronous face requests, ordered completed face results, and automatic decoder replacement.
- MPS operations share a process-wide lock because concurrent PyTorch GPU work caused a native Metal error during the first deployment attempt.
- The launch agent starts after login and restarts the service if it exits.

Final live checks showed both views processing roughly 11 frames per second, frame ages around 0.2 seconds, MPS detection, CoreML face analysis, and no decoder restarts during those checks. These figures are observations for the current scene, not promised crowd throughput.

## What the captured test established

The two-minute recording contains an earlier entry, then the requested exit and return. Four offline combinations—YOLO26n or YOLO26s with ByteTrack or TrackTrack—each found two entries and one exit when all sampled observations were processed. The larger detector did not improve the count on this clip, so the faster nano model was selected for the live workload.

The first real-time replay exposed two issues that offline model-only tests missed:

1. A completed face result could be overwritten by the next empty result before the body loop consumed it. Completed results now use an ordered bounded queue and are drained together. Source timestamps and drop counters are reported separately from body-frame timestamps.
2. The fast entry changed body tracker ID near the gate. Recovery now requires a short, motion-consistent trajectory plus learned appearance and color agreement. Ambiguous alternatives and opposing movement are rejected. A regression test reproduces the recorded positions.

After those fixes, real-time replay counted all three passages. The returning passage was linked to Visitor 005 using the facing camera's confirmed face identity and cross-camera body evidence. CPU/CoreML face embeddings agreed at approximately 0.9997 cosine on sampled frames, so changing the recognizer or relaxing its identity threshold was unnecessary.

Body appearance remains the less certain signal. Measured genuine cross-view scores in this recording were below the original 0.85 setting. The provisional station configuration uses a 0.65 appearance threshold, 0.15 competing-candidate margin, at least three observations per view, and a maximum one-second gap between overlapping track lifetimes. One accepted return match scored approximately 0.652; another replay remained unresolved. That is insufficient evidence for general identity accuracy. Uncertain links stay unresolved while the physical passage stays counted.

## Audited recovery of the original test

The earlier missed entry was recovered from the recording. The existing return entry received the accepted replay identity evidence. Original live evidence was retained inside the audit metadata, and the database was backed up before reconciliation. The dashboard marks these rows as reviewed recordings; they are not presented as successful real-time recognition by the older code.

Other live passages and saved face profiles were preserved. Dashboard totals can therefore exceed the three passages in the validation clip.

## Verification and remaining limits

The full Python suite passed 173 tests before the final presentation-only marker, and the four web packaging tests passed. Coverage includes bounded capture, timestamp gaps, fast crossings, turnbacks, ambiguity rejection, identity persistence, pause/reset behavior, async delivery, model fallback, and MPS serialization. Frontend builds succeeded and the running dashboard was inspected.

The recorded test establishes a concrete improvement on this doorway and these passages. It does not establish crowd-scale precision, recall, demographic performance, or reliable identification of every exit. A broader labeled set with pairs, opposite-direction traffic, similar clothing, children, and occlusion is still needed. Public pretrained weight terms also remain separate from code licenses and must be resolved before commercial rollout.

## Reproducible tools and recovery

- `tools/capture_validation.py`: bounded private recording of both views with authoritative capture-clock sidecars.
- `tools/benchmark_live_models.py`: local model/tracker comparisons with optional truth labels and capture timestamps. Missing truth produces unavailable accuracy metrics rather than invented precision or recall.
- `tools/replay_live_validation.py`: real-time replay through the live asynchronous pipeline, with an isolated copy of the face gallery and database.

Edith keeps the private recordings, reports, source backup, database backup, camera settings backup, and reconciliation audit under the project data and deployment directories. Credentials are excluded from reports and command arguments. No Git push or new hardware purchase was performed.
