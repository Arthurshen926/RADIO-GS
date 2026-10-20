# ADR0038: Test completed-reference evidence across all authorized source views

2026-10-18. N15 is fixed after N14 development results, before N15 prediction.

Keep N14 source-quality reference selection, frozen teachers/geometry, the old
observed Gaussian domain, 120 Huber iterations and 0.5 readout. Independently
transport either the original reference mask or the selected completed mask to
EVERY original authorized source frame with the same two-frame SAM3 path. The
reference frame uses its binary seed directly. Negative object-presence outputs
are unknown, not background. Source evidence is probability/confidence area
integration at 512x384, matching the established diagnostic grid. Target evidence
and rendering remain native 1008x756. Do not change prompts or select task winners.

Six arms: v10, quality_image, quality_native (exact N14 controls),
legacy_source_native, completed_source_native, completed_joint_native (primary).
Source-only fields exclude all target image evidence. Joint fusion assigns equal
total weight to the source group and target: normalize each valid source view by
its confidence mass, average valid views, and independently normalize the target.
No weight search. Solve the three controls jointly as independent columns of the
same fixed operator; this shares computation, not evidence between experiments.
Check normalized operators against the original renderer on every view.

All 48 predictions precede either GT convention. Report IoU/AP, source-response
consistency, and foreground repair versus damage across all eight tasks. Other
four benchmarks remain hash-verified references: no fresh all-five claim.

Stopping rule: if this fixed family fails to improve mean IoU without reducing
mean AP against N14, do not tune weights, view subsets or scene switches on these
scores. Keep N14 and stop this optimization branch. A failure establishes a local
limit of this intervention, not an upper bound for NVOS or the architecture.

Continuation storage remains 8MiB total / 2MiB run / 1MiB artifact. New run <=1MiB;
prediction bank <=256MiB; per-task temporary source evidence <=256MiB; no new
weights or durable dense arrays. At most 16GiB of sparse source operators in RAM.
