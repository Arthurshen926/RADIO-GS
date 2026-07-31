# Easy3D / AGILE3D protocol audit

Date: 2026-07-31

## Scope

This is a checkpoint-only baseline reproduction audit. It does not retrain
Easy3D or modify RADIO-GS.

- official Easy3D commit:
  `b3f5bd70defaa9a601edb0975802775b056c784a`
- official checkpoint SHA256:
  `4a13d16ba2f2470031287812dbbdf1ec6aa14097cb3738e0fe596bb708dc475f`
- official AGILE3D commit:
  `b73638da41edbabe52a1b578d52ddeb8fa552173`
- ScanNet40 release: 312 scenes, 10,357 objects, 5 cm voxels
- paper IoU@1/2/3/5/10:
  `0.682 / 0.746 / 0.773 / 0.796 / 0.817`

## Preprocessing contract

The Easy3D paper says voxel RGB is averaged, but the released `VoxelDataset`
uses duplicate indexed assignment. Main-process multithreaded assignment is
not stable. The actual four-worker training path has one CPU thread per worker
and is stable last-write.

Formal inference reads immutable arrays produced inside that untouched worker
path. The 312-scene cache manifest SHA256 is
`39035ec87a3ff73bd9cfd6eec9a93182b7ebd7d9b2e84515b1c0e51cad453d23`.
A full hash and presence audit found all 10,357 objects in both point and
voxel labels.

## Paired interaction-contract pilot

The same 113 objects from voxel-count p10/p50/p90 scenes were evaluated with
the official checkpoint, batch size 4, BF16, and 10 clicks:

| Contract | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 | Mean absolute paper gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `agile3d_release` | 0.6800 | 0.7368 | 0.7657 | 0.7889 | 0.8102 | 0.650 pp |
| `easy3d_released_code` | 0.6799 | 0.7338 | 0.7600 | 0.7800 | 0.7970 | 1.266 pp |

All 226 contract/object trajectories completed without failure.
`agile3d_release` is selected as the sole formal contract because its mean
absolute gap over all five paper IoUs is smaller. The frozen decision source
is
`output/protocol_audit_20260731/easy3d_agile3d_pilot3_protocol_decision.json`.

## Formal status

Attempt 001 completed 68/312 scenes and 2,676/10,357 objects with zero object
failures before GPU0 was lost. No formal aggregate is claimed. The completed
prefix diagnostic is explicitly non-comparable:

| Interrupted prefix | Objects | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complete release keys | 2,676 | 0.70164 | 0.75922 | 0.78339 | 0.80626 | 0.82681 |
| legacy-key intersection | 2,608 | 0.70159 | 0.75901 | 0.78336 | 0.80644 | 0.82686 |

These rows are a lexicographic hardware-interrupted prefix, not a
representative pilot and not a paper reproduction number. Attempt 002 is
prepared to resume only exact-provenance complete scene shards after GPU0
recovers.

Failure and resume provenance:
`output/protocol_audit_20260731/easy3d_agile3d_formal_agile3d_release_v1/formal_attempt_001_failure.json`.

Partial diagnostic:
`output/protocol_audit_20260731/easy3d_agile3d_formal_agile3d_release_v1/partial68_hardware_interrupted_diagnostic.json`.
