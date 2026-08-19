# Source-only SAM-mask-aligned language teacher sparse sentinel

The attachment's mask-aligned teacher was previously only partial: the dense
crop-summary teacher was not aligned to object masks, and the SAM proposal
path did not emit official text-compatible descriptors for both isolated
region appearance and surrounding context.

`build_sam_mask_aligned_language_teacher.py` now implements the missing
precursor contract. It validates a pre-registered source-only RGB list, the
root official-SAM generation contract, every source-image SHA, every proposal
payload SHA, packed-mask geometry, boxes, areas, confidence, and the frozen
RADIO checkpoint before loading the model. For each proposal it emits:

- a tight masked crop descriptor with fixed gray outside-mask fill;
- a 1.5x expanded unmasked context-crop descriptor;
- frozen 1536D official SigLIP2 crop-summary descriptors for both views;
- SAM quality, stability, seed, candidate, box, and area metadata;
- geometric part-of and shared-direct-parent sibling candidates.

The topology records are explicitly candidates, not semantic truth. A sibling
candidate must share a direct geometric parent and have IoU at most 0.05; a
part-of candidate requires at least 0.95 child containment and child/parent
area ratio at most 0.8.

## Figurines sparse sentinel

The first pre-registered source view, `frame_00001`, passed the fully bound v4
input preflight. Five SAM proposals produced two `[5,1536]` float16 descriptor
tensors. Descriptor norms remain approximately one. The paired masked/context
cosine mean is 0.68915, confirming that the two teacher views are materially
different rather than duplicated. Three geometric part-of candidates and zero
shared-parent sibling candidates were present.

This is only a contract and materialization sentinel. The v4 input is a sparse
single-image grid12 proposal cache, not the requested automatic multiscale
hierarchy. Zero sibling candidates on this frame means it cannot validate
sibling supervision. No benchmark labels, masks, vocabulary, evaluation RGB,
or text query were opened, and no benchmark metric or promotion claim follows.

The next required step is to replace the sparse proposal source with the full
query-independent multiscale hierarchy, then construct cross-view associations
and train/gate proposal-level identity and topology supervision.
