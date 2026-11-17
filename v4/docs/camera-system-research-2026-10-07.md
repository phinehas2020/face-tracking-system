# Research and upgrade plan for the Edith doorway cameras

Research cutoff: October 7, 2026. Prepared for Phinehas Adams.

The installed detector belongs to a current model family, but the deployed system is not yet a reliable two-camera counting and recognition system. The immediate gaps are camera coverage, counting geometry, processing throughput, and association between a face observed in one view and a crossing observed in another. Replacing the detector alone would leave those gaps intact.

The recommended direction is a calibrated doorway counter with independent face recognition and explicit cross-camera association. Keep YOLO26 as the baseline, benchmark TrackTrack and Apple acceleration, and compare stronger face models on the same labeled doorway footage. Treat NVIDIA multi-view systems as a second hardware path if the required accuracy and throughput exceed what the M4 can sustain.

## What is actually installed

Live inspection of Edith confirmed an Apple M4 with 16 GiB of memory. The environment contains Ultralytics 8.4.153, PyTorch 2.14.0, torchvision 0.29.0, OpenCV 4.14.0.94, ONNX Runtime 1.30.0, and InsightFace 2.0. These are installed package versions, not evidence that their newest models or acceleration paths are selected.

| Component | Current behavior | Consequence |
| --- | --- | --- |
| Person detection | `yolo26n.pt`; inference size defaults to 640; no device override | Current detector family, smallest variant, default CPU selection on this Mac |
| Body tracking | ByteTrack plus custom histogram and overlap continuity | No learned body appearance model in the live association path |
| Face recognition | `buffalo_l`, detector input 1280, explicit CPUExecutionProvider | Older model pack running entirely on CPU despite available acceleration options |
| Camera streams | Two 1920×1080 streams configured at 10 fps | Stream rate is higher than the observed rate of analyzed frames |
| Analysis throughput | Roughly 4–5 processed frames per second per view in earlier sustained checks | A fast crossing may have only a few usable observations |
| Crossing rule | Side-view box bottom center; vertical line at 50%; three stable observations per side | Sensitive to clipped bodies, short visibility, and misplaced line |
| Front and side connection | Shared template and label dictionaries; separate body tracks | Does not transfer a front-view identity to a side-only body |
| Recovery | Separate decoder processes, ten-second stall recovery, launch agent | Feed stability improved and decoder replacement was tested; passage accuracy remains separate |

Evidence pointers: [live processing](../backend/ingress/live.py), [camera coordination](../backend/ingress/dual_live.py), [model adapters](../backend/ingress/vision/adapters.py), [face aggregation](../backend/ingress/face_registry.py), and [track continuity](../backend/ingress/vision/continuity.py).

Before the latest diagnostic restart, both views had detected a person, but no passage was recorded. In the subsequent diagnostic window, the side view had no body tracks; the facing view had three tracked observations and no usable faces. The observed bodies were approximately 50–94 pixels wide in the original 1920-pixel image. Those samples do not establish that the complete walk was captured, so they cannot isolate the full cause of the miss.

Both cameras advertise resolution options through 3840×2160 in their authenticated streaming capabilities, while the selected mode is 1920×1080 at 10 fps. Frame-rate options extend to 20 fps, but supported resolution/rate combinations need verification before changing them. Testing a higher-resolution facing stream with native-resolution head crops could improve usable detail; decoding a larger frame and immediately shrinking the entire image to the same network input would not preserve that benefit. No camera-resolution change was made during this research.

The inspected side image shows the counting line over the wall beside the door and excludes the floor. The configured footpoint is therefore often the bottom of a clipped bounding box rather than a visible foot. This needs a physical placement or explicitly different counting-anchor decision before geometry can be trusted.

## Person detection and tracking

| Candidate | Release and evidence | Decision |
| --- | --- | --- |
| YOLO26 | Official January 14, 2026 release; end-to-end detector with supported CoreML export | Retain as baseline. Compare `n` and `s` on doorway recall and latency. [Release](https://www.ultralytics.com/news/ultralytics-redefines-state-of-the-art-vision-ai-with-yolo26) |
| RF-DETR | Released transformer detector and ICLR 2026 work; official CoreML export is experimental | Strong detector challenger after export parity and M4 timing checks. [Repository](https://github.com/roboflow/rf-detr), [CoreML](https://rfdetr.roboflow.com/latest/exports/coreml/) |
| DEIMv2 | September 2025 paper and released weights | Useful offline accuracy challenger; no established turnkey M4 advantage. [Paper](https://arxiv.org/abs/2509.20787), [code](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| YOLO27 | Official preview states weights, code and package support are unreleased | Exclude from deployable shortlist. [Release status](https://docs.ultralytics.com/models/yolo27) |

ByteTrack is an efficient 2022 baseline, not the newest tracker. TrackTrack, from CVPR 2025, is the first practical challenger to test; OC-SORT and FastTracker are useful lower-cost comparisons, while Deep OC-SORT adds appearance processing. Edith already contains configuration files for all six supported trackers, including BoT-SORT. A package upgrade is not required simply to try them. Their tradeoffs are documented in [Ultralytics tracking guidance](https://docs.ultralytics.com/modes/track).

TrackTrack's original results used a different detector and appearance stack, so selecting its configuration does not reproduce the paper's accuracy automatically. [Original implementation](https://github.com/kamkyu94/TrackTrack).

BoostTrack and BoostTrack++ are released alternatives, but their benchmark workflow includes interpolation or postprocessing, and their published setup targets NVIDIA hardware. FDTA is a genuinely newer CVPR 2026 research candidate, but its compiled components and depth inputs make it a larger integration project. Neither is the first fix for a doorway that is barely observed. [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack), [FDTA](https://github.com/Spongebobbbbbbbb/FDTA).

At the current 640-wide input, a 50–94-pixel body in a 1920-wide frame becomes approximately 17–31 pixels wide. A fixed doorway crop can preserve more useful detail before resizing, provided it retains approach and departure space and maps detections back into full-frame coordinates. Full-image tiling is a fallback with additional inference cost. The underlying crop-and-resize approach is supported by [SAHI](https://arxiv.org/abs/2202.06934); its benefit here still needs measurement.

## Face recognition and aggregation

| Candidate | Why it matters | Deployment judgment |
| --- | --- | --- |
| Current buffalo_l | SCRFD detector and ResNet50 recognition provide an established baseline | Keep for controlled comparisons. Upgrading the Python package does not replace these weights. [Model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) |
| AdaFace | CVPR 2022 quality-aware recognition, explicitly evaluated on difficult low-quality imagery | First practical recognition challenger. Published comparisons do not establish superiority over the exact installed buffalo checkpoint. [Repository](https://github.com/mk-minchul/AdaFace) |
| AdaFace with KP-RPE | CVPR 2024 landmark-aware position encoding addresses alignment, scale and pose | Stronger accuracy candidate if its latency is acceptable. [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.html), [models](https://github.com/mk-minchul/CVLface) |
| LVFace | ICCV 2025 Highlight; official ONNX and PyTorch checkpoints released | Offline challenger. No verified real-time M4 result; the largest published variant does not win every metric. [Repository](https://github.com/bytedance/LVFace) |
| QCFace | WACV 2026 Oral with released checkpoints and explicit recognizability modeling | Current research candidate. Its repository notes an unvalidated IJB-S evaluation path; Mac throughput and export need testing. [Repository](https://github.com/hpcc-hcmut/QCFace) |
| NAM/LPM/QGP | September 2026 low-quality recognition work | Watchlist only: official code and checkpoints still listed as coming soon. [Paper](https://arxiv.org/abs/2609.01014), [release status](https://github.com/candllq/nam) |

The stronger architectural change is to preserve a face track through short misses and combine several trustworthy observations. Current face continuity depends heavily on overlapping boxes; at sparse frame rates, a moving face may acquire a new track before it accumulates the required samples. Use motion and appearance evidence to preserve continuity, retain sharp aligned crops, reject conflicting samples, and aggregate embeddings with quality weights. CAFace provides research evidence for video-level aggregation, but its trained aggregator is tied to a particular backbone and cannot simply be attached to buffalo embeddings. [CAFace](https://github.com/mk-minchul/caface).

Do not lower identity thresholds simply to make the dashboard show names. Measure false matches, missed matches, and unresolved rate separately. Every model change requires a versioned gallery and recalibrated thresholds; embeddings from different recognizers are not interchangeable.

InsightFace 2.1 was released October 3, 2026. Its runtime supports CoreML for face analysis, but the application's explicit CPU provider overrides automatic selection. Its new PersonAnalysis feature is not a complete replacement: it supports CPU/CUDA, returns current-frame matches, and does not provide persistent tracks or history. [Release history and feature scope](https://pypi.org/project/insightface/), [runtime](https://github.com/deepinsight/insightface/blob/master/python-package/docs/runtime.md).

The newly named Raccoon and Cheetah packs do not establish a new identity backbone: `raccoon_l` and `cheetah_l` use the same `w600k_r50.onnx` recognition model as `buffalo_l`. Cheetah adds body detection and body ReID, while Raccoon changes packaging and supporting tasks. Renaming the selected pack would not by itself upgrade identity accuracy. [Official PersonAnalysis model contracts](https://github.com/deepinsight/insightface/blob/master/python-package/docs/person_analysis.md), [model guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md).

Native face detail remains essential. AWS recommends at least a 50×50 native-pixel face for its 1080p face-search setup and explicitly warns that upsampling cannot supply missing detail. That is a vendor-specific guideline, not a universal guarantee. An initial engineering target here is roughly 80–112 pixels of face width in the useful recognition zone, followed by measured validation. Repositioning, suitable optics and exposure matter more than synthetic face restoration. [Camera guidance](https://docs.aws.amazon.com/rekognition/latest/dg/recommendations-camera-image-video.html).

## What modern multi-camera systems actually do

The relevant frontier is explicit spatial and temporal association, not a shared face lookup table.

| System | Current evidence | Fit for this project |
| --- | --- | --- |
| GaugeAlign | Winner of AI City 2026 Track 1; frozen TAO Sparse4D backbone with camera/world alignment work | Strong evidence for geometry and system integration. NVIDIA research stack, not an M4 drop-in. [Code](https://github.com/HyperbolicCurve/GaugeAlign-AICity2026) |
| Syn2RealTrack | August 2026 release and AI City runner-up; detector, pose, visible body-part appearance and calibrated global matching | Useful design reference for front/side fusion and rejecting ambiguous matches. Heavy stack; no demonstrated real-time M4 performance. [Code](https://github.com/SKKUAutoLab/aic26_mc3dp) |
| GMT | CVPR 2026 global trajectories and joint association across views | Research comparison; official setup uses CUDA and Detectron2. [Code](https://github.com/FoxCanned/GMT) |
| DeepStream 9.1 MV3DT | Calibrated, timestamp-aligned multi-view tracking with global identities and later corrections | Complete engineering reference if moving to NVIDIA hardware; not native to Edith. Requires suitable overlapping views. [Documentation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html) |

Use final organizer results when discussing the 2026 competition: GaugeAlign/EVA scored 56.5447 3D HOTA and SKKU/Syn2RealTrack 48.2336. The runner-up authors also publish an earlier 52.0118 figure; it differs from the reviewed result. None of these numbers is doorway counting accuracy. [Final verified results](https://www.aicitychallenge.org/2026-challenge-winners/).

For body appearance, OSNet is a practical lightweight baseline from 2019, not cutting edge. KPR/SOLIDER is the more relevant occlusion-aware challenger: body-part visibility helps avoid comparing hidden regions. KPR remains a component of a 2026 competition system despite originating in 2024. CLIP-ReID and TransReID are established transformer comparisons. SapiensID is newer CVPR 2025 research, but its noncommercial weight terms and unknown M4 throughput make it an evaluation candidate rather than the default deployment choice. [OSNet](https://github.com/KaiyangZhou/deep-person-reid), [KPR](https://github.com/VlSomers/keypoint_promptable_reidentification), [SOLIDER](https://github.com/tinyvision/SOLIDER), [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [SapiensID](https://github.com/mk-minchul/sapiensid).

SAM 3.1 is another real 2026 advance, but its advertised 32 fps video result is on an H100. It detects and tracks prompted objects; it does not establish that an attendee is the same person across doorway cameras or repeat visits. It is more useful here as an offline annotation or difficult-occlusion experiment than an assumed M4 live replacement. [Meta's release](https://ai.meta.com/blog/segment-anything-model-3/).

## Recommended architecture

1. **Observe the crossing area.** The side view must show people before and after the threshold. If a footpoint is the anchor, feet must be visible. Otherwise choose and validate an upper-body anchor explicitly. Position the facing view for readable faces rather than a broad room overview. [Axis counting guidance](https://help.axis.com/en-us/axis-p8815-2-3d-people-counter).
2. **Create a passage independently of identity.** Use ordered outside, transition and inside zones, direction, trajectory intersection, hysteresis, and one event per completed passage. A person turning around at the threshold should not become a completed entry.
3. **Preserve uncertain passages.** Store the passage immediately with identity unresolved. Keep a bounded window for later evidence; failing to read a face must never erase an otherwise valid crossing.
4. **Bind face to body within the facing view.** Use same-frame spatial association, persistent tracks and multiple consistent face samples.
5. **Associate bodies across views.** Combine valid camera transitions, capture timing, direction, geometry where available, and several body appearance observations. Use one-to-one matching and ambiguity margins. Similar clothing or the nearest timestamp alone cannot establish identity.
6. **Attach or revise identity with evidence.** Keep the original passage and the reason for association. Allow corrections without changing the physical count. A face directed toward one camera during entry may face away during exit; counting and identifying departures are separate capabilities.
7. **Schedule work by urgency.** Decode continuously, give body tracking a predictable frame budget, and perform more expensive face/re-identification work on selected crops asynchronously. Keep bounded buffers and discard stale inference.

Do not apply a ground-plane homography to the bottom of an arbitrary cropped torso. The present side view needs reframing before floor geometry is a defensible association signal.

## Hardware and model licensing

Edith already has the hardware path for an accelerated baseline. Compare explicit MPS against CPU for detection and CoreML against CPU for face analysis. CoreML can use eligible Apple compute units; MPS uses the GPU. Fixed input shapes and warmed caches matter. Report actual execution providers, export parity, sustained processing rate, frame age and latency while both cameras run. Published CUDA/TensorRT or H100 numbers do not predict this Mac's result. [Apple MPS](https://developer.apple.com/metal/pytorch/), [ONNX Runtime CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html), [YOLO CoreML](https://docs.ultralytics.com/integrations/coreml).

A short benchmark on Edith used the existing YOLO26n checkpoint, one 1920×1080 camera snapshot, `imgsz=640`, three warmup calls and ten measured calls per backend while the live station continued running. CPU median prediction latency was **33.14 ms** (28.95–36.11 ms); MPS median was **12.73 ms** (11.54–14.84 ms), approximately **2.60 times faster** for this detector call. MPS was synchronized around timing. This is a static-scene microbenchmark, not a sustained two-camera or face-recognition benchmark and not an accuracy comparison. Face inference, decoding and coordination still contribute to total latency. The live backend was not changed by the benchmark.

Live capability checks also returned `MPS available=True`, `CoreMLExecutionProvider` in ONNX Runtime's provider list, and `cpu` from Ultralytics' default device selector.

No specific GPU purchase is justified until these measurements and the required number of cameras are known. NVIDIA is a credible path for heavier multi-view processing if local acceleration is insufficient; cloud or remote inference would also introduce transport latency and a separate decision about sending video off the premises.

Model weight terms are distinct from code licenses. InsightFace's public pretrained packs are designated noncommercial research unless separately licensed; its code is MIT. LVFace also restricts downloaded models despite MIT code. KPR uses a Hippocratic license, and SapiensID is CC BY-NC 4.0. Check the exact candidate weights and training-data terms before using them for business operations. AuraFace is an Apache-2.0 model alternative according to its publisher, but its published evidence does not establish superiority for this doorway. [InsightFace licensing](https://www.insightface.ai/solutions/face-recognition-licensing), [LVFace terms](https://github.com/bytedance/LVFace), [KPR license](https://github.com/VlSomers/keypoint_promptable_reidentification/blob/main/LICENSE), [SapiensID license](https://github.com/mk-minchul/sapiensid/blob/main/LICENSE), [AuraFace model card](https://huggingface.co/fal/AuraFace-v1).

Commercial face engines are another comparison path. Use the exact evaluated algorithm versions in NIST's one-to-many identification evaluation to screen vendors, then test the offered SDK on this doorway's video and operating threshold. A strong still-image rank does not establish live crossing recall or cross-camera association. [NIST FRTE one-to-many](https://pages.nist.gov/frvt/html/frvt1N.html).

## Evaluation and implementation order

The first acceptance gate is a normal single-person entry and exit, reliably counted with a trace explaining each result. The latest attempt has not passed that gate.

Next create synchronized, labeled clips for at least 50 entries and 50 exits, including fast walks, hesitation, turnbacks, two people together, opposing traffic, similar clothing, partial occlusion, and decoder reconnection. Keep a development set separate from the final evaluation set. Record capture times, model versions, thresholds, camera calibration and hardware backend so comparisons are reproducible.

Measure separately:

- Passage recall, false counts, duplicate counts and wrong direction.
- Identity-link precision, unresolved passages, false merges and false splits.
- Per-view track fragmentation and usable-face coverage.
- Sustained analyzed fps, p95 frame age, recovery duration and CPU/GPU/memory use.

A 99% precision/recall goal can guide engineering, but 100 passages cannot establish high statistical confidence at that level. Do not report a zero-error small trial as certified event-scale accuracy.

Implement in this order: coverage and geometry; measured acceleration; track continuity and independent passage events; front-to-side association with abstention; controlled model comparisons; sustained crowd and recovery validation. Select the detector, tracker and recognizer that win those tests together. No universal leaderboard currently proves one complete stack is best for these two cameras.

The research did not install new models, purchase hardware, or establish that the missed walk-through is fixed. The existing feed repair and count reset are separate completed operations.
