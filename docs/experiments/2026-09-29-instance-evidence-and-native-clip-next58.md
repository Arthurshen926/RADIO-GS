# Instance evidence and native CLIP exchange — next58

## Outcome

Seven full-four-scene controls completed. There is no substantial gain, and no
replacement of the retained next56 research candidate. The best new selected
macro is47.958 versus47.863 (+0.095pp), requiring an additional identity encoder;
this is not a breakthrough or established independent-test improvement.

| Control | 2D IoU | AP | Selected projection IoU |
|---|---:|---:|---:|
| Retained next56 |58.087|74.965|47.863|
| Per-instance local SigLIP identity |58.087|74.965|47.871|
| Local identity + instance area balancing |58.087|74.965|47.845|
| Compatible enclosing proposal censors negatives |58.087|74.965|47.816|
| Local OpenCLIP identity instead of SigLIP |58.087|74.965|47.958|
| Native OpenCLIP field + source scalar calibration |1.415|64.623|47.863|
| OpenCLIP field + scope + prior confidence distribution |54.793|71.446|47.863|
| Native OpenCLIP field + generic-negative readout |35.691|60.830|47.863|

Percent, scene macro. Selected projection is the frozen selected-Gaussian alpha
rendering evaluated against2D annotations, not independent physical3D GT. All
results are repeatedly used development scenes. No new ScanNet/NVOS/UQIS result.

## Instance identity versus union confidence

The preceding decoder uses the union of within-view detections and the maximum
identity confidence among those detections. This loses instance-level reliability.
For instance k, c_k = detector_score * sigmoid(foreground_context_margin / RMS).
The new positive confidence at a pixel is max_k(mask_k*c_k)/max_k(mask_k),
normalized by the original frame maximum so the outer confidence multiplication
preserves the intended absolute weight. Qualified negative evidence remains as
before; lower identity confidence does not certify background completeness.

The local control retains the union target but weights its positive loss spatially;
it does NOT infer cross-view physical identity or retain a learned object inventory.
The second arm further balances instance masks by inverse area. All start from
the same next54 membership and run the same120-step next56 decision objective.
Thus the near-zero differences are not confounded by an extra continuation pass.

Multiple detections are not rare:20.0/45.1/22.9/52.4% of detected query-views have
more than one instance for figurines/ramen/teatime/waldo respectively. Nevertheless,
these particular weighting changes barely affect measured retrieval. This does
not prove that correct instance association is unnecessary; confidence weighting
alone does not supply it.

The extent control reuses next49's source-only native-compatible containment
relations and original source SAM regions. Only negative labels inside a compatible
enclosing proposal become unknown. No new positives, target masks, alias rules
or proposal selections based on benchmark scores are introduced. Applied to the
stronger decoder, this is slightly worse than next56; not retained. The combined
identity-plus-extent arm is implemented but was NOT run or claimed effective.

## Reference implementation transfer

The local VALA implementation uses OpenCLIP ViT-B-16 with laion2b_s34b_b88k,
224-pixel resize and CLIP normalization. Source: `/root/baselines/VALA/eval/openclip_encoder.py`.
Weights already existed locally. No upstream code was changed, no new download
or baseline reproduction is claimed.

First exchange only the foreground/context identity encoder. Keep the same source
RGB, detections,1.5x windows and union decoder; encode raw runtime text with the
matching OpenCLIP text tower. Recompute margins and RMS normalization. Full4
selected47.958 is only0.095pp above next56; figurines regresses slightly. This does
not justify adding an encoder to the mainline for a claimed major gain.

Then rebuild a512D query-independent OpenCLIP region field on exactly next52
geometry, using the same source frames, saved region indices and mixture pooling
as next54. Foreground crops reconstruct masks by upsampling saved low-resolution
SAM masks, with a3-pixel box margin and neutral127 background. Original SigLIP
teacher crops used original-resolution masks. Therefore the exchange changes
both teacher/capacity (512D vs1536D) and crop-mask precision; it is not a pure
encoder ablation and says nothing universal about which VFM is better.

Query-independent region features are fused by alphaT contributions. Source-only
scalar calibration uses the same240-step recipe as next54. Scope fusion retains
existing source SigLIP identity evidence to isolate the field exchange; weighted
quantile calibration retains the same next53 confidence reference. The full field
candidate is not four-teacher retraining or a complete standalone compact model.
Actual selected membership stays at next56 throughout these semantic-field arms.

Direct scalar-calibrated CLIP outputs have near-zero IoU on two scenes and macro
1.415 despite64.623AP: this recipe produces a severe fixed-threshold mismatch.
It is not evidence that the teacher recognizes no targets. Adding source scope
and old mass gives54.793/71.446, below retained58.087/74.965.

The additional native readout follows VALA's query-versus-generic-negative rule:
`sigmoid(10*(cos(query)-max cos(object, things, stuff, texture)))`.
The four negatives are fixed independently of benchmark query vocabulary.
It bypasses source scalar calibration and yields35.691IoU/60.830AP. This fixes
some threshold behavior but does not recover retained ranking quality. This arm
uses the saved FP16 field rather than the construction-time FP32 field; tiny
precision differences should not be attributed solely to the scoring rule.

## Scientific interpretation

No claim that identity/extent is solved is supported. Source-local weighting,
unknown enclosing ranges and a different native encoder are insufficient in these
implementations. The source masks can still identify the wrong object and use
incompatible contents/container extents; neither encoder confidence nor generic
negative contrast establishes the intended relation. Do not advertise tiny deltas
or extra foundation-model calls as the paper's contribution.

Retain next56. The unresolved intervention must produce and validate explicit
entity identity AND requested extent, beyond weighting the same ambiguous masks.
Its value must be demonstrated on held-out task definitions and with matched
source/teacher budgets. The current online detector + field + confidence prior
pipeline still has a fragmented method identity; these controls do not turn it
into a finished offline four-query system. No CVPR-ready or SOTA claim is made.

## Artifacts

* `output/instance_evidence_next58/summary.json`
* `output/instance_evidence_next58/source_multiplicity.json`
* `scripts/research/decode_instance_evidence_next58.py`
* `scripts/research/build_clip_identity_next58.py`
* `scripts/research/run_instance_evidence_next58.py`
* `scripts/research/rebuild_clip_field_next58.py`
* `scripts/research/refine_clip_scope_next58.py`
* `scripts/research/align_clip_scope_next58.py`
* `scripts/research/run_clip_field_next58.py`
* `scripts/research/read_clip_contrastive_next58.py`
* `scripts/research/run_clip_contrastive_next58.py`
* `scripts/research/summarize_instance_evidence_next58.py`

All started jobs completed. No defensive tests or audit framework were added.
Frozen source caches, geometry, annotations and evaluator remain unchanged.
