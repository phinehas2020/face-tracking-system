# Where the Fair counter loses people

## What the recordings support

The booth Mac retained ten full-resolution Station 2 stills (five paired moments, two cameras) and 80 seconds of timestamped side-camera boxes and clothing-color histograms. The original 2025 camera videos live on the Extreme Pro drive. The stills support person-detection evaluation. The cached boxes support controlled tracker interruptions. Counting errors on natural crossings, and the motion scheduler’s effectiveness during a sustained crowd, require the original videos.

## People visible in the stills

I drew approximate visible-person boxes before running inference. The scoring region was the central foreground (box center at x 96–864, y 259–540 in a 960×540 rendering). Boxes outside the region were also annotated where needed to handle boundary matches. One-to-one assignment maximized eligible matches at a minimum intersection-over-union of 0.50. The confidence cutoff was 0.25 for all four detectors. The first three paired moments contain 39 eligible boxes. The last two contain 33. Both groups come from the same cameras and visitors can recur. Each label represents one appearance in one frame.

| Detector | First six frames matched / 39 | Last four matched / 33 | Unmatched in-region predictions on last four |
| --- | ---: | ---: | ---: |
| YOLO26n, 960 | 35 | 29 | 4 |
| YOLO26m, 960 | 36 | 31 | 2 |
| YOLOv8m, 960 | 35 | 32 | 3 |
| RF-DETR-Large, native 704 | 38 | 33 | 8 |

The extra predictions include duplicate or differently bounded boxes. Single-reviewer annotations of occluded bodies introduce uncertainty. Raising RF-DETR’s confidence cutoff to 0.50, which gave the highest development F1 among five tested settings, gave 30/33 matches and zero unmatched predictions on the last four frames. RF-DETR’s authors report current real-time detector results on COCO. The figures here come from running the released model locally on the Fair images. [RF-DETR paper](https://openreview.net/pdf?id=qHm5GePxTh)

At 14:24:08 on the side view, two adjacent people behind a foreground visitor failed the nano detector’s box match. Both medium YOLO and RF-DETR matched them separately. At 19:22:48 on the facing view, the three YOLO models missed a child mostly concealed by a stroller canopy. RF-DETR matched that child at 0.25 and lost the match at 0.50. This is a concrete cost of filtering the extra RF-DETR boxes by confidence alone.

## Track continuity through a controlled loss of observations

Each trial starts with an observed person moving near an overlapping box. The person had a cached observation in all 20 frames. Eight consecutive observations are then hidden from the trackers while all neighbors remain present. Recovery requires the earlier tracker ID to reappear on the target within four following frames, with no other ID on the target in that interval and no assignment of the earlier ID to a neighbor during the gap or recovery. Missing pre-gap IDs remain failures. Every trial also runs with the observations intact. The 68 challenges represent 43 source track IDs.

| Tracker association | Prior ID retained / 68 | Prior ID assigned to a neighbor |
| --- | ---: | ---: |
| Booth TrackTrack configuration | 41 | 2 |
| Standard TrackTrack settings | 32 | 1 |
| ByteTrack | 45 | 3 |
| OC-SORT | 48 | 5 |
| FastTracker | 43 | 3 |
| Booth configuration with looser lost-track matching | 49 | 6 |
| Booth configuration with corroborated handoffs | 43 | 2 |

These are association-only runs of the installed Ultralytics implementations. Each receives the same previously tracked boxes. Image-based camera motion, detector-rescued proposals and learned ReID are absent. Original cache IDs provide proxy correspondence. People missed entirely by the earlier detector cannot enter the trials. The two temporal halves were inspected during development. The observations support failure diagnosis, not a public-benchmark ranking. TrackTrack was selected for its published crowded pedestrian results. [TrackTrack paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)

## Replay change and remaining limits

The optional `crowd-consensus` mode runs the booth’s TrackTrack association, ByteTrack and OC-SORT on the same detector output. A newly numbered primary track can inherit the former ID only when both supporting trackers agree, the pair had three consistent observations, its last observation belonged to the missing primary track, and the torso-color histograms agree across a gap of at most 1.2 seconds. A primary ID already visible in the frame takes precedence. The corresponding `crowd-control` mode uses the exact same detector path and booth tracker configuration on its own.

This fixed two induced breaks in the same cached trajectory, starting at frames 8852 and 9068. Continuity elsewhere was unchanged across the 68 trials. The first half tied the booth baseline at 19/29 and the later half improved from 22/39 to 24/39. OC-SORT retained more trajectories overall and gave the old ID to a neighbor more often. A separate full-video integration replay on the noncrowded booth walkthrough confirmed two entries and one exit with YOLO26m in both control and candidate modes (498 detector frames, motion-triggered schedule).

From `v4/`, use the existing `tools/benchmark_live_models.py` flags with `--tracker crowd-control --tracker crowd-consensus` for the paired comparison. It also accepts `--tracker ocsort` and `--tracker fasttrack`. The controlled-gap evaluation accepts a local JSONL or gzipped JSONL file. Private inputs, labels, per-trial outcomes, the scorer, and the side-by-side case image for this run are under `data/crowd-benchmark/`.

A passage-level evaluation still needs direction and timestamp labels from the continuous fair clips, frozen before comparing model runs. A child in a stroller also requires an explicit counting convention for any attendance total. These experiments did not change either booth’s live settings.
