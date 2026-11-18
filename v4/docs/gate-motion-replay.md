# Motion-triggered doorway replay

A camera can keep delivering frames while the body detector is busy. The side camera can use three narrow strips of pixels to locate the few seconds that deserve full-rate inference. Every reported passage still has to pass the existing tracker and trajectory counter.

## The walkthrough

The October 7 booth capture has 2,381 timestamped side-camera frames over 118.3 seconds of coverage. Its documented sequence is entry, exit, entry. The pixel analysis independently identified activity at 23.7–26.6, 38.1–41.0 and 44.6–47.3 seconds. Inspection of the recorded views confirmed the order. Directions were inferred using pixels at the existing gate. All tracker runs used YOLO26n at 960, TrackTrack and MPS.

| Detector schedule | Frames sent to detector | Entries | Exits |
| --- | ---: | ---: | ---: |
| Every tenth frame | 239 | 0 | 0 |
| Evenly spaced, lower matched budget | 498 | 0 | 1 |
| Every tenth frame plus motion bursts | 498 | 2 | 1 |
| Every fourth frame | 596 | 0 | 1 |
| Evenly spaced, higher matched budget | 811 | 2 | 1 |
| Every fourth frame plus motion bursts | 811 | 2 | 1 |
| Every second frame | 1,191 | 2 | 1 |

At the lower matched budget, the tracker confirmed entry at 24.32 s, exit at 39.54 s and entry at 45.26 s with bursts. The evenly spaced control confirmed only the exit at 39.76 s. At the higher matched budget the strategies tied. These comparisons use one development clip and one uniform sampling phase per budget. Frame counts measure detector calls after warmup. Decoding, pixel analysis and plotting also take time.

## How it works

1. Sample 80 locations on the calibrated gate and on each of two parallel lines, four percent of normalized image coordinates to either side. Keep only positions visible on all three lines.
2. Estimate each pixel’s usual color from up to 401 timestamp-spaced frames in each continuous section. Mark activity when at least 20% of samples on each strip differ from the background by more than 32 levels in a color channel. Join short interruptions and split analysis at capture gaps over 250 ms.
3. Compare color changes at neighboring strips over delays of up to half a second. The order suggests direction when both pairs independently agree. Simultaneous changes, weak matches, long activity and truncated windows can yield an unresolved direction.
4. Run the detector at its baseline stride. Include every available frame within each motion window and one second on either side. Every window gets this treatment, including unresolved directions. Replay in chronological order with one tracker so a burst can use the approach and departure.
5. Compare the resulting tracker events with each window. Review motion-only windows, opposing directions, multiple tracker events in a window, and tracker passages outside all windows. A window can contain several people or a swinging door.

The image under each example plot is a photo finish. Each column shows what the center line saw at that instant. The JSON includes every window and the source hashes. The figure shows up to three examples with disagreements first.

## Reproduce

From `v4/`, use the vision environment and a side-camera trajectory calibration in the same format as the existing model benchmark. The timestamp JSONL supplies the capture clock.

```bash
python tools/audit_gate_motion.py --video side.mp4 --timestamps side.jsonl \
  --calibration gate.json --output motion

python tools/benchmark_live_models.py --video side.mp4 --timestamps side.jsonl \
  --calibration gate.json --model models/yolo26n.pt --imgsz 960 \
  --tracker tracktrack --device mps --stride 10 --max-seconds 120 \
  --motion-audit motion/audit.json --output burst.json
```

For the equal-budget control, repeat the second command with `--uniform-control --output uniform.json`. To draw the comparison, repeat the first with `--benchmark uniform.json --output control-review`. The audit checks the video, timing file and calibration against the benchmark. Each report gets a new directory.

## What would justify a live trial

The current prototype uses background samples from the whole recorded segment and one second of pre-roll. Live use would need a causal background estimate and a bounded source-frame buffer. A dropped camera frame is unavailable to either method. Dense groups, alternating directions, moving doors, long occlusions and camera movement can defeat the strip signal.

The next experiment is a blind review of busy fair footage with frozen settings. Label directions and timestamps for every crossing in selected clips, including quiet intervals. Compare burst and uniform schedules at equal budgets across multiple starting phases. Measure directional misses, extra passages, reviewer seconds per hour and end-to-end cost including the pixel pass. The two cameras still need a separate test for assigning an identity to the correct passage.

Line sampling into a time image follows earlier line-of-interest counting research. The use of inexpensive visual evidence to allocate heavier inference also has precedent in edge video analytics. The experiment here evaluates that combination at this doorway. [Ma and Chan, CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ma_Crossing_the_Line_2013_CVPR_paper.html) · [Li et al., SIGCOMM](https://web.cs.ucla.edu/~harryxu/papers/li-sigcomm20.pdf)
