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
tracker with the RF-DETR second look and a 640 px detector (commit b1ded1b).
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

- With the second look on, the side camera's detect-and-track step averaged
  328 ms a frame while a person was in view, and a track was present in only 26
  frames across those four visits. The second look runs RF-DETR on the CPU, about
  300 ms, inside the counting loop whenever the gate strips move.
- With it off the same step averaged 131 ms and one of two crossings was
  counted; the miss was a person carrying a box on one shoulder.
- At 17:41 UTC the booth was returned to the tracker file and 960 px detector it
  had before that change (`live_tracker.yaml`, no second look), keeping the
  crossing zone and activity capture on. The earlier settings and status
  snapshots are in `deployment/activity-20261118/`.
- **The restored configuration is not yet shown to count.** Nobody had walked
  through in the first ten minutes after the change. Ten crossings read by one
  person from strips are enough to see the collapse, not to rank the trackers.
- The 12 windows saved before 17:41 UTC were cut with an earlier version of the
  window boundaries, where consecutive strips overlapped. Leave them out:
  open the review page with `?since=2026-11-18T17:41`, and pass the same value
  to `/api/live/scorecard` or `ingress scorecard --since`.

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
  wait was not established. With the restored 960 px detector the same step
  takes about 160 ms and the wait is 1 ms, about 6 frames a second per camera.
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
