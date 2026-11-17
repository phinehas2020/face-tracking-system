# Global identity resolution, 2026-11-14

The live pipeline decides who someone is once, at crossing time, against one mean
template per person. Anything it cannot decide then is lost: an unresolved passage
kept no face evidence. This change keeps the evidence and adds a second, global
pass that treats identity as entity resolution over the whole event.

## What was added

| Piece | File | Purpose |
|---|---|---|
| Tracklet evidence | `backend/ingress/evidence.py`, `tracklet_evidence` table | Face samples for every finished tracklet, enrolled or not |
| Resolver | `backend/ingress/resolution.py` | Constrained clustering, review queue, derived counts |
| Attendance range | `backend/ingress/estimation.py` | Hard bounds plus a conditional estimate with an interval |
| Metrics | `backend/ingress/clustering_metrics.py` | Pairwise and B-cubed precision/recall, count error |

`FaceRegistry` writes a tracklet when it expires, when tracks are cleared, and at
shutdown, but only for a registry created with a camera id. `LiveSession` passes
one; the paired replay's per-camera control registries do not, so they are
unchanged. A storage failure increments `evidence_errors` and never interrupts
counting.

```bash
.venv/bin/ingress resolve <event-id>            # report only
.venv/bin/ingress resolve <event-id> --apply    # also write links and the review queue
.venv/bin/ingress calibrate-resolution <event-id>
```

`GET /api/events/{id}/resolution` and `POST /api/events/{id}/resolution/apply`
do the same and first flush tracklets that are still on screen.

## Method

1. **Nodes.** Each enrolled identity (template, confirmed variants and its
   tracklets) is one group. Each unenrolled tracklet with at least two samples
   and internal consistency of 0.45 or more is its own group.
2. **Constraints.** Two tracklets of the same kind that overlap by a second or
   more in one camera are different people, unless they are near-identical
   (0.75 or more), which means one person tracked twice. A human "different"
   decision is also a cannot-link.
3. **Scores.** Cosine similarity becomes log-odds through a two-parameter
   logistic calibration.
4. **Clustering.** Groups merge greedily by mean log-odds, highest first. An
   automatic merge needs a mean probability of 0.97 or more and no pair below
   0.25 similarity. Human "same" decisions are applied after the evidence pass,
   so a confirmation can only add merges.
5. **Three-way decision.** Cluster pairs between 0.30 and 0.97 go to
   `identity_links` with status `review`. The existing decision endpoint works
   on them, and each decision becomes a constraint on the next run and a
   labelled pair for calibration.
6. **Counts.** A passage belongs to a person through its online identity or,
   failing that, through its own track's stored tracklet. A cluster counts as a
   person only if it contains an enrolled identity or three samples, the same
   bar as online enrollment. Nothing in `passages` is rewritten.

The attendance report gives a lower bound (distinct resolved people), an upper
bound (that plus every unresolved entry), and, when at least 20% of entries are
resolved, an estimate assuming every entry is equally likely to yield a face
match. That assumption fails when the same people are systematically unreadable,
so the estimate is conditional and the bounds are not.

## Sources

- Constraints from tracking: Kalogeiton and Zisserman, *Constrained Video Face
  Clustering using 1NN Relations*, BMVC 2020.
- Clustering on signed pairwise costs: Tang et al., *Multiple People Tracking by
  Lifted Multicut and Person Re-identification*, CVPR 2017.
- Automatic / review / reject bands: Fellegi and Sunter, *A Theory for Record
  Linkage*, JASA 1969.
- Why the review band matters: Debnath et al., *UAC: An Uncertainty-Aware Face
  Clustering Algorithm*, ICCVW 2021, report common clustering algorithms
  predicting up to 9.6 times the true number of identities on noisy embeddings.

## Verified, and not

- 197 backend tests pass, nine of them new. They cover a synthetic crowd with
  look-alike relatives and fragmented tracklets, a simulated review loop,
  cannot-link, duplicate tracks, identity merges, passage attachment, a human
  overrule surviving a re-run, registry persistence, purge, the estimator
  against a simulated crowd, calibration fitting, and the API.
- On synthetic crowds of 45, 450 and 1,687 people the resolver made no false
  merges and ran in under a second. Synthetic embeddings are far cleaner than
  real faces; this shows the algorithm is correct, not that it is accurate here.
- **Not run on real footage.** No tracklet evidence exists yet; it starts
  accumulating when this code runs on the station.
- **The calibration is a prior** (even odds at 0.45, 0.97 at 0.60). The report
  says `calibrated: false` until reviewed pairs replace it.

## Known limits

- Side-camera passages attach to facing-camera faces only through the existing
  online cross-view link, which requires an enrolled identity.
- No image is stored for an unenrolled tracklet, so a reviewer cannot yet see
  those faces. The web console does not show the new queue.
- The pairwise matrix is dense; it refuses more than 6,000 nodes.
- A tracklet keeps the registry's last 8 to 12 samples, not a chosen best set.

## Measurements on the booth Mac, 2026-11-14

Read from the running station and from a copy of its database. The deployed code
and service were not changed.

| Measure, 2026-10-08 to 2026-11-14 | Value |
|---|---:|
| Entries counted | 1,141 |
| Entries linked to a face identity | 19 |
| Entries whose crossing track never had a face | 1,049 |
| Face profiles enrolled / linked to any entry | 41 / 9 |
| Cross-camera links made / expired unlinked | 14 / 2,213 |
| Facing-camera faces accepted / rejected on pose | 9,392 / 179,037 |
| Side-camera faces accepted / rejected on pose | 2,150 / 55,364 |

- The 41 enrolled profiles are mutually distinct: the resolver found no merge
  and no review candidate among them. Duplicate profiles are not the problem.
- Of the last 64 unlinked crossings, 42 failed the body-appearance threshold
  (scores 0.55 to 0.65 against a threshold of 0.65, usually with only one
  candidate in view), 18 passed appearance but had no face identity, and 4 were
  ambiguous. Only those 64 are retained in memory, so the full distribution is
  unknown.
- A staged copy of this code replayed the 2026-10-07 two-camera recording. A
  control run of the unchanged deployed code gave the same frame counts, pose
  counts and passages, so the change is behaviour-neutral there. Both runs
  processed about 2.8 frames per second against 6 on 2026-10-07 and counted no
  passages, because the live service was using the machine at the same time.
- From that 32-second window the resolver took 14 real tracklets, attached 6 to
  enrolled profiles, and queued 10 pairs for review. There is no ground truth
  for those links.
- Head pose on the facing recording (57 faces): 61% pass the 25/20 gate, and all
  failures are yaw. That recording is a walk-through by someone looking at the
  camera; live traffic passes 5%. Similarity by angle was inconclusive: too few
  faces, and a second person in frame.

## Added after those measurements

- **Decision record.** Every crossing that ends unlinked now keeps the final
  reason, appearance score, margins and best facing track in
  `passages.evidence.cross_view_rejected`, so thresholds can be set from the
  full distribution instead of the last 64.
- **Identity checked last.** The cross-camera matcher now runs every appearance
  test before asking for a face identity. `face_identity_pending` therefore
  means the body match was accepted. Link outcomes are unchanged.
- **Resolver uses those matches.** A crossing rejected only for
  `face_identity_pending` is attached to the stored faces of that facing track.
- **Single-sample tracklets admitted.** Face tracks on the facing camera
  fragment to one sample each at low frame rates. A cluster still needs three
  samples or an enrolled profile to count as a person.
- **Pose histograms.** `runtime.face_pose_gate` reports yaw and pitch buckets
  for every detected face, accepted or rejected.

202 backend tests pass.

## Real-footage tests on the 2025 fair recordings, 2026-11-14

Run on the development Mac (M3 Pro), where the recordings drive is mounted and
nothing competes for the machine. Private outputs are in the git-ignored
`data/resolver-crowd-test/`.

### Crowd replay with the resolver

The three 45-second crowd excerpts from `recording-catalog/crowd-replay.json`,
every frame processed (2,025 frames).

| | Online pipeline | After global resolution |
|---|---:|---:|
| Entries counted | 36 | 36 |
| Entries linked to a person | 6 | 9 |
| Distinct entrants | 3 | 5 |
| Unresolved entries | 30 | 27 |

- All 248 stored tracklets came from the Station 1 counter view. The Station 2
  wide and approach views produced none: every face there was too small or
  failed the pose gate. 27 of the 30 unresolved entries are on tracks that never
  had a face, so no identity method can resolve them. In the counter view,
  linked entries went from 5 of 9 to 8 of 9.
- Face tracks fragment badly: 138 of the 248 tracklets hold a single sample.
- With the default prior the review queue hit its cap of 200 for one 45-second
  clip. That prior is too cautious for this footage (see below).
- None of these links has ground truth.

### What different people score: same-frame pairs

`tools/score_survey.py` scores every pair of faces detected in the same frame.
Those are different people by construction, so this is an exact impostor sample.
Eight minutes of counter-view footage at about 2 frames per second:

| Pose of the less frontal face | Pairs | 99th percentile | Maximum | Pairs at 0.40 or more |
|---|---:|---:|---:|---:|
| Inside the 25/20 gate | 564 | 0.241 | 0.395 | 0 |
| Up to 35 degrees | 1,390 | 0.388 | 0.591 | 13 |
| 35 to 50 degrees | 1,954 | 0.312 | 0.490 | 3 |
| Beyond 50 degrees | 2,255 | 0.259 | 0.442 | 1 |

A pair that persists across frames is counted each time, so 13 pairs may be one
or two pairs of people standing together.

- A two-component mixture fitted to unlabelled tracklet scores put the
  same-person boundary near 0.30. The 0.395 impostor is about five standard
  deviations outside the fitted different-person mode, so that model's tail is
  wrong and it would merge different people. It was removed.
- Fitting instead to these impostors plus same-tracklet sample pairs gives even
  odds at 0.29 and automatic merge at 0.43. With it, clusters fell from 153 to
  80 and the review queue from 200 to 104; linked entries stayed at 9. It is
  available as `ingress calibrate-resolution <event> --impostor-scores <file>`
  and reported as `method: constraints`, not as calibrated. It is not the
  default: 564 pairs from one camera cannot bound rarer look-alikes, and the
  margin over the worst observed impostor is 0.04.

### Is the pose gate too strict?

Same scan with the gate lifted. A face that stays in place between consecutive
sampled frames is the same person; each pair below has one face inside the gate.

| Pose of the other face | Pairs | Median similarity | Share at 0.60 or more |
|---|---:|---:|---:|
| Inside the gate | 413 | 0.735 | 82% |
| Up to 35 degrees | 189 | 0.656 | 68% |
| 35 to 50 degrees | 46 | 0.522 | 20% |
| Beyond 50 degrees | 27 | 0.455 | 0% |

- The gate is about right for this recognizer. Beyond 35 degrees the same person
  rarely reaches the 0.60 match threshold.
- Faces up to 35 degrees keep most of their signal and would nearly double the
  usable faces (959 to 1,811 in this scan), but their impostor tail is heavier.
  They are candidates for supporting evidence at a stricter threshold, not for
  enrollment.
- In this view 54% of detected faces exceed 20 degrees of pitch, so camera
  height costs about as much as head turn.
- The same two tables are a label-free benchmark for comparing recognizers:
  a pose-robust model should raise the genuine rows without raising the impostor
  rows.

203 backend tests pass.
