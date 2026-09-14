# Text identity and descriptor-order repairs — 2026-09-14

## Confirmed deployed defect: text preprocessing differs from the paired encoder

Freshly reencoded every deployed positive/negative text query using the official SigLIP2 giant checkpoint and C-RADIO canonicalization (lowercase, ASCII punctuation removal, underscore replacement and whitespace collapse). Three deployed embeddings differed materially:

| Scene / query | Old vs canonical cosine | Old vs uncanonicalized cosine |
| --- | ---: | ---: |
| teatime / dall-e brand | 0.880217 | 0.99999994 |
| waldo_kitchen / Stainless steel pots | 0.913621 | 1.0 |
| waldo_kitchen / pour-over vessel | 0.802849 | 1.0 |

All other rows and generic-negative banks agreed with canonical reencoding to numerical precision (minimum cosine approximately 0.99999988). A separate raw-tokenization experiment reproduced the three old rows, confirming the actual preprocessing defect. Waldo's old cache metadata even declared official canonicalization, so the label was insufficient evidence.

The text loader now requires explicit processed-query strings when labels need canonicalization, and rejects zero embedding rows. The standard cache builder records those strings. A complete fresh set of banks is available through the final input manifest below. This is a preprocessing/provenance guard, not a cryptographic proof that arbitrary external embeddings were computed correctly; the real reencoding audit provides the evidence for these particular banks.

The loaded HF checkpoint exposes a 1152-to-1536 text projection while this installed Transformers version initially constructs a 1152-to-1152 head. The existing checkpoint-head restoration helper was used; its restoration succeeded. Initial loader warnings do not mean the measured vectors used the randomly initialized incompatible head. Fresh canonical and raw reproduction results above were obtained after restoration.

## Actual native-resolution impact

Reevaluated both affected scenes on GPUs 2/3 for text scoring and CPU for surface projection. Kept membership, geometry, source-resolution repair, selector, generic negatives and pixel threshold 0.2 fixed. Rendered native masks and used the same verified public mask scorer.

| Measurement | Before text repair | After text repair |
| --- | ---: | ---: |
| teatime scene mIoU | 0.07495009 | 0.07495328 |
| waldo_kitchen scene mIoU | 0.12299605 | 0.12616953 |
| dall-e brand observation-mean IoU | 0.01419998 | 0.01429402 |
| Stainless steel pots observation-mean IoU | 0.15532187 | 0.18522031 |
| pour-over vessel observation-mean IoU | 0.02494815 | 0.06486628 |

These are development evaluations. Only the two affected scenes were rerendered this round; no new four-scene headline score is claimed. The actual improvement is modest and cannot explain or resolve the entire remaining low accuracy.

## Independently reproduced descriptor-order defect

`_load_fragment_prototypes` sorted manifest frames by frame number but paired the concatenated features with token IDs stored in mapping-view order. Nonchronological source selection could therefore silently associate valid descriptors with the wrong objects while all total dimensions matched.

It now reconstructs descriptor order from the state's source-frame sequence, rejects duplicate/mismatched frame inventory, checks each payload's frame identity and proposal count, and rejects zero/nonfinite descriptors. A two-frame nonchronological reproduction test verifies the descriptor-to-token mapping; it would fail under the old sort. Per-frame checks also prevent compensating count errors from passing a total-row check.

The current four LERF states happen to use chronological source frames, so this ordering repair does not explain their existing scores. Independently compared every retained object prototype with its referenced source crop: figurines 1,616; ramen 1,078; teatime 1,552; waldo_kitchen 1,068; total **5,314**, all maximum absolute error **0**.

## Inputs and validation

Use `/mnt/pool/sqy/results/RADIO-GS/output/v4_repaired_query_inputs_20260914/manifest.json` for subsequent runs. It binds all four previously sealed resolution-corrected scene states/hypotheses to freshly verified positive and negative banks. New banks are built from explicit fields instead of inheriting stale derived hashes or an old frozen-bank origin after changing embeddings.

Evidence:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_text_pairing_audit_20260914/report.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_text_pairing_audit_20260914_verified/report.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_text_pairing_repair_20260914/teatime/report.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_text_pairing_repair_20260914/waldo_kitchen/report.json`

Regression after loader/ordering changes: **275 passed, 2 skipped**. Metadata finalization does not change numerical tensors. Final caches were cold-loaded and checked against the evaluated fresh embeddings. Historical artifacts are preserved; old banks requiring normalization without explicit processed-query provenance now intentionally require regeneration rather than silent reuse.

## Remaining work

The current map still has severe identity-ranking and object-extent errors. The validated fixes do not support claiming that further numeric thresholds will restore historical quality. Next priority is an object-equal separation of correct identity retrieval, wrong merging and missing extent on the same repaired map, while tracing actual negative/unknown evidence and completion inputs. Use counterexamples and real outcomes to distinguish implementation defects from an inadequate association model.
