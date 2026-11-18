# Activity windows and the counting scorecard, 2026-11-18

Nothing the station counts had been checked against what happened: a live
passage kept no picture, so there was no way to say how many people it misses.
This change keeps a short film strip of every active moment at the door and
turns human answers about those strips into a scorecard.

## What was added

| Piece | File | Purpose |
|---|---|---|
| Activity windows | `backend/ingress/activity.py`, `activity_windows` table | A strip per camera whenever people or gate motion are present on the counting view, counted or not |
| Review page | `backend/ingress/review.html`, `/api/live/review` | Play both views, state how many people really went in and out |
| Scorecard | `/api/live/scorecard`, `ingress scorecard <event>` | Found and correct rates with 95% Wilson intervals, misses, extras |
| Stage timings | `runtime.stage_ms`, `runtime.fps_with_people` | Where a frame's time goes, split by whether anyone is in view |

Turn capture on with `"activity_capture": true` in the camera file
(`"activity_capture_max_mb"`, default 600, bounds the strips).

## Method

1. **Window.** A window opens when the counting view has a tracked person or
   frame-to-frame change on the three gate strips, and closes after one quiet
   second or at ten seconds. Passages recorded inside it are attached.
2. **Strip.** Each camera keeps its last 16 seconds of encoded frames at six a
   second. A closed window becomes one contact-sheet JPEG per camera (up to 40
   frames, 480 px wide), with one second of context before and after. A window
   cut at ten seconds and the one that continues it share no frames, so no
   crossing is shown twice.
3. **Uncounted windows are the point.** A window with no passage is saved too,
   at most 30 an hour. Someone standing still is one window; a later stretch of
   the same visit is kept only if something moved at the line. These are where
   missed people are found.
4. **Label, blind.** The reviewer counts crossings of the line inside the shaded
   span of the clip. The station's own count and its passage marks stay hidden
   until the answer is saved.
5. **Score.** Misses and extras are summed per window and direction. An extra
   and a miss of the same direction inside one window cancel, so both figures
   are floors.
6. **Bounds.** Strips live in the event directory and go with an event purge.
   Nothing is written with less than 2 GB free. Past the byte budget the oldest
   strips are deleted; rows and labels stay.

Passage-level labels with a frozen review protocol follow line-of-interest
counting evaluation (Ma and Chan, *Crossing the Line*, CVPR 2013). Intervals are
Wilson score intervals (Wilson, JASA 1927).

## Also changed

- **Empty views are scanned for faces twice a second, not every frame.** The
  side camera had run 325,891 face scans for one detection.
- **Empty face scans no longer fill `tracking_history`.** They pushed every real
  track out of the 240-entry history within about 30 seconds.
- **The resolver counts a face sample once.** The live registry stores a sample
  under its body track and again under its face track; 44 such pairs made 43 of
  the 203 automatic merges and counted double toward the three-sample bar.

## The first thing it found: the station had stopped counting

On the evening of 2026-11-17 the booth was switched to the `crowd-consensus`
tracker with the RF-DETR second look and a 640 px detector (commit b1ded1b),
and its detector model from `yolo26n` to `yolo26m` in the service definition.
Counted passages between 13:00 and 18:00 UTC, the booth's morning:

| Day | Passages counted |
|---|---:|
| Weekdays 2026-11-09 to 2026-11-17 | 9, 23, 42, 44, 45, 51, 58 |
| 2026-11-18 | 3 |

The first hour of activity windows shows why. Read by eye from the strips:

| Configuration, 2026-11-18 | Real crossings seen in strips | Counted |
|---|---:|---:|
| Crowd-consensus, second look on (17:18 to 17:28 UTC) | 8 | 0 |
| Crowd-consensus, second look off (17:30 to 17:32 UTC) | 2 | 1 |
| `live_tracker.yaml`, 960 px, still `yolo26m` (17:41 to 18:16 UTC) | 2 | 0 |

- With the second look on, the side camera's detect-and-track step averaged
  328 ms a frame while a person was in view, and a track was present in only 26
  frames across those four visits. The second look runs RF-DETR on the CPU, about
  300 ms, inside the counting loop whenever the gate strips move.
- With it off the same step averaged 131 ms and one of two crossings was
  counted; the miss was a person carrying a box on one shoulder.
- At 17:41 UTC the tracker file and 960 px detector from before that change
  were restored (`live_tracker.yaml`, no second look). That was not enough:
  with `yolo26m` at 960 px the detect-and-track step took 212 ms a frame with a
  person in view, about 3 frames a second, and two more crossings were missed.
- At 18:16 UTC the model was set back to `yolo26n`. In the next 23 minutes all
  five activity windows carried a counted passage (3 in, 2 out), at 6.1 frames a
  second with a person in view. **Those five were not checked against the
  strips**, so this shows counting resumed, not that it is accurate.
- The crossing zone and activity capture stayed on throughout. Earlier settings,
  the earlier service definition and status snapshots are in
  `deployment/activity-20261118/`.
- Crossings above were read by one person from strips: enough to see the
  collapse, not to rank trackers or models.
- Label from the restored setup onward: open the review page with
  `?since=2026-11-18T18:17`, and pass the same value to `/api/live/scorecard` or
  `ingress scorecard --since`. Earlier windows belong to other configurations,
  and the first 12 were cut with overlapping strips.
- The side camera also sees people inside the office through its doorway and
  the interior window. Three of eight windows in one stretch were that, with no
  crossing.

## Measurements on the booth Mac, 2026-11-18

Read from the running station before the first deploy unless noted.

| Measure | Value |
|---|---:|
| Entries counted since 2026-10-08 | 1,280 |
| Entries linked to a person, online / after resolution | 20 / 24 |
| Frames received / processed per second, side | 20 / 7.0 |
| Frames received / processed per second, facing | 15 / 7.6 |
| Facing-camera faces beyond 45 degrees of yaw | 820 of 1,405 |
| Facing-camera faces accepted by the pose gate | 57 of 1,134 |
| Stored face tracklets with a single sample | 284 of 400 |
| Passages linked across cameras since 2026-11-17 23:00 UTC | 0 of 21 |
| Free disk on the booth | 6.4 GB of 228 GB |

- **The facing camera looks through a door leaf.** On 2026-10-07 its view was
  the open doorway. On 2026-11-18 the right leaf stands across the lens, so
  people are seen through glass and muntin bars, close to the camera, in
  profile, with heads cut by the top of the frame.
- **The door swings through the side camera's view.** The inward-opening leaf
  sweeps across the counting line as a person enters or leaves. Whether that
  costs crossings is a hypothesis: every miss seen so far happened under the
  slow configuration above.
- **Where a frame's time goes.** With the 640 px detector and nobody in view, a
  frame took about 48 ms of work and the loop then waited about 80 ms before the
  next frame, although the cameras deliver 15 to 20 a second. The cause of that
  wait was not established. `yolo26m` at 960 px took about 160 ms with a 1 ms
  wait; `yolo26n` at 960 px takes 32 ms on the side camera with nobody in view.
- **Body appearance is expensive when people are present.** OSNet added 22 to 73
  ms a frame across three short samples (26 to 259 frames).
- **A different detector runtime is not a shortcut.** YOLO26m through ONNX
  Runtime's Core ML provider took 51 ms a frame against 58 ms for PyTorch MPS,
  both with the live service running. A direct Core ML export failed in
  coremltools 9.0 with torch 2.14. Neither was deployed.
- **The crossing zone is still learning:** 7 entry and 8 exit samples of the 20
  needed. Facing-image x ranges from 0.21 to 0.51 across them, so the fitted
  zone will be wide.

## Verified, and not

- 250 backend tests pass on the booth, ten of them new: ring thinning, a
  counted window with a strip from each camera, a loitering visitor, movement in
  a later stretch, the hourly limit, low disk, pruning with labels kept, gate
  motion, the scorecard arithmetic with a start time, and the review API.
- Capture is running on the booth and saved real windows. The review page was
  loaded in Chrome through the tunnel: both views play in step, the counting
  line is drawn, a label saved and the scorecard updated. That test label was
  cleared afterwards.
- **No window is labelled, so there is no accuracy figure yet.**
- The idle face-scan change was not shown to raise frame rate: 7.0 before, 7.3
  after, within noise. It applies to the counting camera only.

## Known limits

- A window is labelled as a whole. Which of two people went which way is not
  recorded, so identity links are not scored yet.
- Motion is frame-to-frame on processed frames only; a person the detector
  misses entirely while moving slowly may not open a window.
- The interior window behind the counting line shows the next room, so movement
  there can open motion-only windows.
- Capture stops, and counts the skip in `runtime.activity_capture`, when less
  than 2 GB of disk is free. The booth has about 6 GB.

## First measured result (labelled 2026-11-18)

85 clips from 18:17 to 21:27 UTC were labelled blind, on `yolo26n`, 960 px,
`live_tracker.yaml`, line at x=0.55.

| | counted | true | missed | extra |
|---|---|---|---|---|
| in | 28 | 30 | 2 | 0 |
| out | 25 | 31 | 7 | 1 |

Recall 85% (95% interval 74–92%), precision 98% (90–100%), 75 of 85 clips exact.

What the ten wrong clips show:

- Eight misses are one person passing close to the camera, head and shoulders
  at the bottom of the frame. Running `yolo26n` over the saved frames finds
  them at 0.86–0.90 confidence in every frame, so detection is not the cause.
- On exits the box centre is last seen at x≈0.45–0.48, then the person is
  behind the door. With the line at 0.55 that is one observation past the
  line, and the counter needed two.
- Two labels look doubtful and were left as labelled: `act-98c56411` (the
  clip shows a person leaving; labelled 0, counted 1 out) and `act-fbef7ebe`
  (labelled 1 out; the person stays inside the office).
- Not explained: `act-c2b510e6` and `act-6200af45` have several clear
  detections on both sides of the line and were still missed live. No track
  positions were stored, so the cause is unknown.

Replaying the counter over detections from the saved frames (a simple IoU
tracker, not the live one; it agrees with the live count on 77 of 85 clips):

| line | single clear observation confirms | right | missed | extra |
|---|---|---|---|---|
| 0.55 | no | 53 | 8 | 0 |
| 0.55 | yes | 57 | 4 | 0 |
| 0.57 | yes | 58 | 3 | 0 |
| 0.60 | yes | 59 | 2 | 2 |
| 0.68 | no | 57 | 4 | 13 |

These numbers are fitted to the same 85 clips, so they are a prediction, not
a measurement.

## Change deployed 21:36 UTC

- Line moved to x=0.57.
- `confirm_distance` 0.07: one destination observation at least that far past
  the line confirms a crossing that intersected the gate and moved
  `min_motion`. Unset keeps the two-observation rule.
- Each saved clip now has `<id>-trace.json` beside its strips: the tracker's
  anchor positions `[offset, track, x, y]`, so the next unexplained miss can
  be read off.

Label the new setup from `?since=2026-11-18T21:36`. Watch `extra` as well as
recall: a single-observation confirm has less protection against one box
jumping across the line.
