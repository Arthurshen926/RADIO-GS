# Confidence-preserving support-function repair with unchanged tensors

Fixed before this arm's benchmark predictions. Development cohort, not a new
independent final test. Follow ADR0014/0015; do not promote a task winner bundle.

The preceding full study established zero NVOS relation activation and nineteen
inactive correct UQIS point supports. It did not uniquely distinguish weak
coordinates, unknown dilution, averaging and scale gates. This intervention
repairs their association composition as one declared package; it cannot isolate
each subchange's causal effect.

For support component i and region r, read soft positive p and exclusion n from
the SAME completed soft-reference tensors. k=p+n is known mass. Admit compatible
regions only when p>n (positive majority CONDITIONAL on known evidence); absent
evidence and a tie abstain. Use p=k*(p/k) as within-view preference, retaining regional known mass. Normalize preferences within
each view and multiply by the maximum admitted k in that component/view, capped
at one. Finally mix these per-component associations using normalized input
weights. Thus unknown components cannot veto known ones, weak evidence is not
amplified into full votes, and alternative rays retain their input mass. This prevents a near-zero regional observation from borrowing the confidence
of another region in its view. Duplicate
copies of all masks in a view do not multiply that view's confidence. Distinct
partial mask confidences need not be calibrated probabilities.

Keep the physical scale gate [0.5,2], target P/N matrices, protected member prior,
finite-evidence update (prior+P)/(1+P+N), signed product, native encoders, candidate
program and typed readouts unchanged. One common member operator serves all
tasks. No training, extra source view, teacher or labelled member head. Copy all
30 parent field.pt payloads byte-for-byte into independent exports; change only
the declared evidence function in their manifests. Retain the full 58.3 GB tensor
cost, including now-unused legacy hard association coordinates. No compression
claim or source bank access at runtime.

Functional tests: unknown versus exclusion, confidence bound, linear mixtures,
duplicate rows/masks, full-scribble chunking and CPU/CUDA parity, saved/reloaded
signed and inferred inputs. A diagnostic six-statistic partial-evidence operator
must reproduce v10 joint_anchor_evidence in the binary, complementary-visible-
exclusion, one-anchor limit, including nested disagreement and missing views.
It is NOT yet a transferred supervised decision head and does not change this
arm's readout. Soft partial evidence is not complete v10 visibility.

Before prediction, replay two cases per valid source view: an exact point and
two full projection rays with .25/.75 weights (all observed contributors). An
independent NumPy/source-raster calculation associates each support component,
combines masks in pixel space and only then lifts target rows. Compare association
and positive/exclusion outputs within 1e-5. Target replay includes 512 uniformly
sampled carrier rows plus every source-query support row, not every Gaussian.
Require nonempty evidence, identical payload hashes and all30 export receipts.

Run all31 prediction jobs before scoring: all four LERF scenes and selected-set
projections, eight signed NVOS cases and actual subsets, ten ScanNet scenes with
19/15/10 categories, nine UQIS scenes/all268 queries/four modalities. No adaptive
thresholds or per-scene selections. Compare soft reference and protected parent;
report paired scene intervals and v10 resource differences. Replay all eight
authorized NVOS prompts and all67 UQIS3D supports, distinguishing admitted views,
evidence strength, selection gains/losses and known-support precision/recall.
Keep native outputs exact and run five typed cold queries from a copied export
with source file access denied. Binary primitive parity, source replay and real
task improvements are three different acceptance claims.

## Source-only correction before scoring

The first mixture function was stopped after construction and partial benchmark
prediction, before any benchmark scoring. Its source-only counterexample is
p=[1,1e-6], n=[0,0], one view: normalizing p/k gave association [.5,.5]. A
vanishing observation borrowed confidence from the strong region. This is a
functional deficiency independent of benchmark outcomes. Preserve all30 v1
exports and partial predictions plus source_counterexample_stop.json in
mixture_functions_v1; do not present v1 as a completed five-task study.

The corrected evidence kind is confidence_mixture_partial_view_evidence_v1. Only
the within-view preference changes from p/k to p; the conditional p>n gate,
component/view confidence, mixture, final P/N update and all other conditions
remain as above. The same test gives [1/(1+1e-6),1e-6/(1+1e-6)], while an isolated
.001 observation still contributes only .001 of a view. Test both conditions,
along with binary v10 parity, full rays, signed inference and CPU/CUDA. Freeze
this correction before ANY v1/v2 benchmark score; all v2 predictions must finish
before scoring. Root confidence_mixture_functions_v1, arm confidence_mixture_function.
