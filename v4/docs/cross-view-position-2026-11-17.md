# Linking the two cameras by position, 2026-11-17

The side camera counts crossings; the facing camera reads faces. A crossing gets
an identity only if it can be put on the right facing-camera person. Until now
that link rested on body appearance (OSNet) between the two views.

## What the booth showed

Forty crossings recorded after the 2026-11-17 deploy, read from
`passages.evidence.cross_view_rejected` and the in-memory decision log
(saved privately as `data/booth-snapshots/status-20261117T2310Z.json`):

| Measure | Value |
|---|---:|
| Crossings linked | 0 of 40 |
| Failed the 0.65 appearance threshold | 39 |
| Appearance score, minimum / median / maximum | 0.48 / 0.56 / 0.66 |
| Best facing track had an enrolled face | 0 |
| Crossing instant inside the best facing track's lifetime | 25 |
| Facing track began 0.1 to 0.8 s after the crossing | 8 |
| Facing track ended 0.06 to 2.3 s before the crossing | 7 |

- The side camera looks across the doorway at close range; the facing camera
  looks through it into the room. A person crossing is side-on and cropped in
  one view and frontal and distant in the other, which is the hard case for
  body re-identification.
- The facing camera loses people at the door frame and the glass, so one
  person is several short tracks. Enrollment needs three consistent face
  samples on one track, and the fragments rarely have them.

## Method

1. **Crossing zone** (`backend/ingress/crossing_zone.py`). A person crossing the
   side line is at a repeatable place in the facing image. For each direction
   the zone is a trimmed Gaussian over the facing box at the crossing: centre x,
   centre y, log height, and the time offset to the nearest facing sample (zero
   when the track spans the crossing). The offset lets a track that starts just
   after the crossing, or ends just before it, count.
2. **Learned, not surveyed.** A sample is taken only when exactly one person is
   in each view and the facing person is moving. Samples are stored in
   `passages.evidence.cross_view_zone` and the most recent 600 are reloaded at
   start, so moving a camera retrains the zone.
3. **Readiness.** A direction is ready with 20 inlier samples and when at most
   half of all observed facing positions fall inside the zone. Both figures are
   in `cross_camera_association.zone` on the status endpoint.
4. **Decision.** Among moving facing people, the nearest to the zone links if it
   is inside the 99% gate, the runner-up is at least 6 further in squared
   distance (a likelihood ratio of about 20), and no other side track claimed it
   within three seconds. Someone standing in view cannot be the link.
5. **Fragments.** Facing tracks are stitched when a later track starts within a
   second of where an earlier one stopped and is its only possible successor.
   The linked person is the whole chain.
6. **Face from anywhere on the chain.** A linked crossing takes the one enrolled
   identity found on any piece of the chain, and waits 30 seconds for it
   instead of 8. Two identities on one chain fail closed.
7. **Resolver.** `ingress resolve` follows a position link (mode `on` only) to
   the stored tracklets of every piece of the chain, so faces that never
   enrolled online still resolve the crossing.

Time alone still never links: before the zone is ready the decision is recorded
as `zone_learning` and nothing is assigned.

## Modes

`"association_zone"` in the camera file: `"off"` (default, behaviour unchanged),
`"shadow"`, or `"on"`.

Shadow learns, decides and records for every crossing but assigns nothing. It
also registers crossings whose side track already has a face. Those are never
linked; they are a labelled check. `zone.label_agree` and `zone.label_disagree`
count how often the position link lands on a facing person with the same
identity. That is the accuracy figure to read before switching to `on`.

## Sources

- Khan and Shah, *Consistent Labeling of Tracked Objects in Multiple Cameras
  with Overlapping Fields of View*, PAMI 2003: the field-of-view line of one
  camera, as seen in the other, learned from lone walkers.
- Makris, Ellis and Black, *Bridging the Gaps between Cameras*, CVPR 2004:
  entry and exit zones and transition times learned without labels.

## Verified, and not

- 220 backend tests pass, 15 of them new. They cover the zone fit and outlier
  trimming, learning before linking, linking with no appearance evidence,
  shadow mode, a known bystander, a person standing in the zone, two people at
  the door together, a follower well behind, stitched fragments, a face read 15
  seconds after the crossing, conflicting faces, the labelled check, storage
  and reload, and the resolver.
- Those tests use simulated box trajectories. They show the logic is right,
  not that the zone is tight enough on this door.
- **Not yet run on live traffic.** The zone has no samples until this code runs
  on the booth.

## Known limits

- Stitching and motion thresholds are starting values, not fitted.
- The online path still needs an enrolled identity on the chain. Pooling face
  samples across fragments happens only in the resolver.
- Chains are not yet a must-link constraint between tracklets in the resolver.
- The labelled check will be small: about 2% of crossings have a face on the
  side track.
