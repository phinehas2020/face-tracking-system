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
   second or at eight seconds. Passages recorded inside it are attached.
2. **Strip.** Each camera keeps its last 13 seconds of encoded frames at six a
   second. A closed window becomes one contact-sheet JPEG per camera (up to 40
   frames, 480 px wide), from 1.5 s before to 1 s after.
3. **Uncounted windows are the point.** A window with no passage is saved too,
   at most 30 an hour, and someone standing in view is one window, not one every
   eight seconds. These are where missed people are found.
4. **Label.** The reviewer gives true entries and exits for the window. The
   system's own count is the default, so a correct window is one keypress.
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

## Measurements on the booth Mac, 2026-11-18

Read from the running station before the deploy unless noted.

| Measure | Value |
|---|---:|
| Entries counted since 2026-10-08 | 1,280 |
| Entries linked to a person, online / after resolution | 20 / 24 |
| Frames received / processed per second, side | 20 / 7.0 |
| Frames received / processed per second, facing | 15 / 7.6 |
| Facing-camera faces beyond 45 degrees of yaw | 820 of 1,405 |
| Facing-camera faces accepted by the pose gate | 57 of 1,134 |
| Stored face tracklets with a single sample | 284 of 400 |
| Crossings linked across cameras since 2026-11-17 | 0 of 60 |
| Free disk on the booth | 6.4 GB of 228 GB |

- **The facing camera looks through a door leaf.** On 2026-10-07 its view was
  the open doorway. On 2026-11-18 the right leaf stands across the lens, so
  people are seen through glass and muntin bars, close to the camera, in
  profile, with heads cut by the top of the frame.
- **The door hides the counting line from the side camera.** In the first
  uncounted window captured after the deploy (17:18 UTC), one person came in,
  walked to the office and left again; the station counted neither crossing.
  The inward-opening leaf covers the line as the person passes it. One window,
  seconds after a restart, read by eye: an observation, not a rate.
- **Frame rate is set by the shared GPU, not by any one model.** After the
  deploy a frame takes about 48 ms of work with nobody in view and the loop then
  waits about 80 ms for its turn. With a person in the facing view the detector
  took 79 ms and body appearance (OSNet) another 73 ms, giving 6.0 frames a
  second over the first 26 such frames.
- **A different detector runtime is not a shortcut.** YOLO26m through ONNX
  Runtime's Core ML provider took 51 ms a frame against 58 ms for PyTorch MPS,
  both with the live service running. A direct Core ML export failed in
  coremltools 9.0 with torch 2.14. Neither was deployed.
- **The crossing zone is still learning:** 7 entry and 8 exit samples of the 20
  needed. Facing-image x ranges from 0.21 to 0.51 across them, so the fitted
  zone will be wide.

## Verified, and not

- 249 backend tests pass on the booth, nine of them new: ring thinning, a
  counted window with a strip from each camera, a loitering visitor, the hourly
  limit, low disk, pruning with labels kept, gate motion, the scorecard
  arithmetic and the review API.
- Capture is running on the booth and saved real windows; the review page
  serves. **No window has been labelled, so there is no accuracy figure yet.**
- The review page was not exercised in a browser.
- The idle face-scan change was not shown to raise frame rate: 7.0 before, 7.3
  after, within noise.

## Known limits

- A window is labelled as a whole. Which of two people went which way is not
  recorded, so identity links are not scored yet.
- Motion is frame-to-frame on processed frames only; a person the detector
  misses entirely while moving slowly may not open a window.
- The interior window behind the counting line shows the next room, so movement
  there can open motion-only windows.
