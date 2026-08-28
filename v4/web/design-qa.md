# Design QA — Event Time Machine

## Comparison target

- Source visual truth: `/workspace/scratch/780f68e6e236/generated_images/exec-2c3e9bb6-ec39-4d03-81b0-bb06326f5eb7.png`
- Browser-rendered implementation: `/workspace/scratch/event-time-machine-verified-20260828.jpg`
- Combined comparison input: `/workspace/scratch/780f68e6e236/event-time-machine-verified-comparison.jpg`
- State: desktop rehearsal, Fusion v4 selected, synchronized cameras, overlays on, case 1 selected, playhead at 32%, replay paused, deterministic fixture data
- Browser viewport / CSS size: `1363 × 936` CSS px
- Device pixel ratio: `1`
- Source pixels: `1487 × 1058`
- Implementation pixels: `1363 × 936`
- Density normalization: the source was aspect-fit to `1363 × 936` with approximately 24 px of neutral lateral matte per side; it was not stretched. The implementation is a direct 1× browser capture.

## Findings

No actionable P0, P1, or P2 differences remain.

- [P3] Historical evidence detail
  - Location: camera overlays and case-inspector imagery.
  - Evidence: the source shows a denser set of pose/keypoint traces and two exact crops of the reviewed attendee. The implementation shows stable track boxes, identity labels, a calibrated commit line, and generated event-camera crops.
  - Impact: no hierarchy or workflow loss in the prototype; real replay ingestion will replace the demo crops with exact tracklet evidence.
  - Follow-up: render optional pose traces and best-frame tracklet crops when real recording output is connected.

- [P3] Secondary transport controls
  - Location: persistent replay footer.
  - Evidence: the source includes extra jump, snapshot, and fullscreen controls. The implementation preserves the core path—play/pause, 1×/8×/32×, sync, overlays, annotate, position, and elapsed time.
  - Impact: no core rehearsal task is blocked.
  - Follow-up: add snapshot/fullscreen only if operators use them during the historical-video labeling pass.

## Required fidelity surfaces

- Fonts and typography: IBM Plex Sans Variable and IBM Plex Mono reproduce the technical grotesk/monospace hierarchy. Display, metric, label, table, and timecode weights remain legible at the dense desktop scale without unwanted wrapping or truncation.
- Spacing and layout rhythm: the final grid matches the source composition—single-row metric header; two dominant lane feeds; stacked Wide/count-curve evidence; model matrix and queue at right; multi-track timeline plus case inspector below; persistent footer. All regions fit the viewport with no overlap or hidden health rows.
- Colors and visual tokens: warm technical canvas, near-black camera wells, blue selected-model/playback state, olive ground truth, red disagreement state, green health/merge state, and thin gray rules map closely to the source. No decorative gradients are used.
- Image quality and asset fidelity: all three visible camera assets are generated raster images with consistent nonprofit-event tent art direction, sharp WebP delivery, measured crops, and no generic placeholders. Phosphor provides icons; Recharts provides charts. There are no handwritten SVGs, emoji substitutes, or decorative CSS illustrations.
- Copy and content: event, date anchor, replay source, unique count, truth, error, model metrics, case IDs, decision reasons, health tracks, and replay labels are coherent standalone product copy and align with the selected concept.
- Accessibility: all core controls are semantic buttons or labeled inputs, camera panels are keyboard selectable, focus indicators are visible, images have descriptive alt text, the purge dialog is labeled and guarded, and active/disabled states are visible.

## Full-view comparison evidence

The final combined input shows equivalent major-region proportions, order, density, visual hierarchy, and semantic color mapping. The implementation keeps the camera evidence dominant, the model/queue decision column compact, and the lower forensic timeline readable at the same time. The 47 px total aspect-ratio matte on the normalized source was excluded from visual findings.

## Focused-region evidence

The camera mosaic and timeline were also inspected as browser clips at native 1× density. The camera clip confirmed sharp crops, readable track labels/timecodes, aligned feed headers, and a contained count curve. The timeline clip confirmed readable passage bars, merge/review strips, all three health bands, divergence curves, playhead, and scrubber. Separate crop files were unnecessary because these details remain readable in the native-size final capture.

## Interaction and runtime verification

- Rehearsal/Live navigation changes selected state.
- Lane camera selection updates the active evidence border.
- Baseline/Fusion/Candidate selection updates the active matrix column.
- Next/previous case navigation updates the inspected identity pair.
- Overlay toggle changed six visible tracking overlays to zero and back after reload.
- 32× playback advanced the visible elapsed time while Pause replaced Play.
- A fixture review decision reduced the queue from three cases to two and updated the unresolved count; reload restored the deterministic default.
- Lifecycle opened the modal; purge stayed disabled until its exact event-scoped confirmation phrase was entered. The destructive action was not executed.
- Browser console check found zero errors originating from the prototype.
- Browser preview used deterministic fixture mode; API, store, vision-contract, portal, fusion-training, replay-manifest, calibration, and purge behavior are covered separately by the Python test suite.

## Comparison history

1. Initial implementation comparison
   - Earlier implementation evidence: `/workspace/scratch/event-time-machine-qa-final-20260828.jpg`
   - [P1] Major-region composition drift: metric cards, equal-height camera row, model cards below the cameras, and a combined review panel did not match the selected dense forensic layout.
   - [P2] The fixed replay footer covered the final camera-health band by approximately 17 px at the target viewport.
2. Fixes made
   - Rebuilt the screen into the source composition: inline metric header, dominant Lane A/B feeds, stacked Wide/count curve, matrix comparison, separate queue, multi-track timeline, separate inspector, and light transport footer.
   - Constrained the grid with `minmax(0, 1fr)`, explicit desktop region heights, and a timeline bottom at `881.6px` above the footer top at `882px`.
   - Added responsive small-screen stacking, memoized stable React regions, and lazy chart chunks without changing the selected desktop design.
3. Post-fix evidence
   - Final implementation: `/workspace/scratch/event-time-machine-verified-20260828.jpg`
   - Final combined comparison: `/workspace/scratch/780f68e6e236/event-time-machine-verified-comparison.jpg`
   - Post-fix review found no actionable P0/P1/P2 mismatch.

## Implementation checklist

- [x] Match selected layout, hierarchy, density, palette, typography, and imagery.
- [x] Keep every core rehearsal and review control interactive.
- [x] Verify default, selected, playing, overlay-off, decision, dialog, guarded, and restored states.
- [x] Check app-origin console errors.
- [x] Build the Sites-ready output and pass its worker tests.
- [x] Preserve remaining differences only as P3 follow-up polish.

## Follow-up polish

- Add optional pose traces and exact tracklet best-frame crops when last year's recordings are mounted.
- Add snapshot/fullscreen transport actions only if the operator rehearsal shows a real need.

final result: passed
