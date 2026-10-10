# Native region query function before spatial integration

Exploratory full four-scene/68-text LERF cohort, same parent fields, sources,
preencoded query fixtures, temperature10 and strict0.5 selection. No new training,
labels, source frames or query-time source access. Prior development GT has been
seen; this is not a blind final evaluation. All predictions before any new scoring.

Five controls: admission (exact retained union-admission baseline), pooled_parts,
pooled_protected, native_parts, native_protected (primary). Pooled averages the
Gaussian prior on Bpositive; native queries exact source ROI semantic descriptors
using sigmoid(10*(cos(query)-maxcos(generic negatives))) before lifting to Gaussians.
For both acceptors: positive margin max(2p-1,0), same finite view-confidence update,
no scale/top-k gate. Native zero descriptors abstain. Each acceptor prepares P/N
ONCE for its protected/unprotected pair. Keep old failed monotonicity audit intact;
new paired controls require exact monotonic scores and unchanged P/N diagnostics.

Compile the native descriptors using the original construction source_records
region indices, verify source hashes/train split, exact existing relation payloads,
view/scales alignment, and add region_semantic inside the same field.pt. No runtime
external region feature bank. Report added and total bytes; this remains a large,
uncompressed capability reference, not an efficiency contribution.

Matched native/pooled comparisons identify regional acceptance conditional on the
fixed candidate family and update, not full instance detection/association. Source
ROI descriptor preservation does NOT transfer v10 query-conditioned detections,
tracks or reliable whole-instance identity. No benchmark labels enter compilation
or inference, no GT candidate selection, no threshold search, no per-scene routing.
Report response, actual-set projection, AP and all scenes; no default promotion
unless actual scope and broad behavior justify it. No unified-five claim.

If actual scope still fails, do not continue acceptance threshold/temperature
search. Revisit candidate coverage and query-conditioned instance correspondence
with separately declared source/diagnostic contracts. Retain all failures.

Pre-score implementation correction: v1 stopped after figurines predictions and
before ANY scoring when FP16 storage was found to determine query arithmetic.
v2 explicitly uses FP32 native cosine/sigmoid, matching the existing semantic
measure function; descriptors retain original storage precision. No labels/scores
inspected, no threshold adjustment. v1 artifacts and abort receipt are retained.
