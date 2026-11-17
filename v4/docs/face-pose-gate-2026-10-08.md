# Near-frontal face evidence, 2026-10-08

The earlier `main_dual_entry_only.py` intended to reject faces beyond 25 degrees
yaw or 20 degrees pitch. It unpacked InsightFace's angles in the wrong order and
allowed missing pose through. The v4 live encoder loaded only detection and
recognition; its landmark-ratio gate applied only below quality 0.7. Large,
confident detections could bypass that gate entirely.

The live encoder now also loads the existing `buffalo_l` `landmark_3d_68` model.
It accepts face evidence only with finite pose estimates within 25 degrees yaw
and 20 degrees pitch, independent of image size or detector confidence. Missing
pose fails closed. The existing small-face checks still apply. Roll remains
handled by face alignment. InsightFace's actual order is **pitch, yaw, roll**, in
degrees, verified against the installed 2.1 source and upstream:

- [Landmark pose output](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/landmark.py)
- [Rotation-to-angle conversion](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/transform.py)

Rejected samples cannot enroll profiles, update face evidence, or replace
thumbnails. A previously resolved continuous body track retains its identity.
Body crossing counts remain independent. No identity threshold or saved identity
was changed by this deployment. The earlier human-confirmed visitor merge is
preserved.

`runtime.face_pose_gate` exposes limits and cumulative accepted/angle-rejected/
missing-pose counts for each camera. Counters describe candidate observations
before overlap deduplication, not distinct people, and reset with the process.

## Verification

- 188 backend tests passed, including high-confidence off-angle rejection,
  correct pitch/yaw order, invalid/missing pose, crop mapping, independent
  passage counts, and persistence of human-confirmed merges. The merge fixture
  now uses distinct observation times instead of racing the millisecond clock.
- On Edith, the updated encoder scanned 16 selected frames from the existing
  facing-camera test recording: 11 accepted observations, 4 angle rejections,
  zero missing-pose rejections. One frame had no eligible accepted observation.
- Median full-frame encoder latency was 66.45 ms; first-call latency was 235.3 ms.
  Detection, recognition, and 3D landmarks reported CoreML with CPU fallback.
  These are sparse recording checks, not a throughput or accuracy benchmark.
- Backup and machine-readable encoder report:
  `/Users/booth2/face-tracking-system/deployment/pose-gate-20261008-121743/`.
- Deployed and restarted Edith's service. Both cameras returned live with ready
  face workers and the 25/20 filter visible. Follow-up showed 175 completed side
  scans and 136 facing scans, no delivery overflow or face errors, and two live
  facing-camera angle rejections. Frame ages were 139/251 ms. Counts remained
  27 entries / 27 exits; the confirmed variant and canonical Visitor 003 remained.

This repairs a concrete sample-acceptance gap; it does not establish a duplicate
identity error rate. The 25/20 limits reuse the earlier intended operating policy,
not a universally calibrated threshold. Stricter acceptance may leave a passage
unresolved when no frontal view is available. A human-labeled walk-through with
frontal and turned-head passes is still needed to measure that tradeoff.
