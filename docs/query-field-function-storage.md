# Function-only storage fixes, 2026-10-13

## What changed

The actual mixture/confidence and native-mode compiler paths now default to the
existing shared-v2 format. They require an explicit store and refuse implicit v1
fallback. Each input field is loaded once per compiler invocation. A second
function/spec using identical tensors creates a separate manifest and state
identity while reusing immutable blobs. Geometry, native codes, evidence, row
contracts and dtype remain part of the complete state. Complete model cost is
reported; marginal new blob bytes are not reported as the whole model size.

- compile_mixture_evidence.compile_field(..., store_root=STORE) changes the actual
  exported evidence kind, not just an unrelated receipt. Original evidence buffers
  are retained exactly, including float64. Confidence and unweighted mixture
  remain different functions with different spec/state identities.
- compile_native_modes.compile_scene(..., store_root=STORE) compiles the native
  modes once and saves mean/measure manifests over the same tensor state.
  Default computation remains CUDA; device='cpu' exists for tiny scientific
  parity fixtures, not an alternate production experiment.
- Explicit legacy_v1=True retains historical full-package serialization. In the
  mixture v1-parent route, the original payload archive SHA is preserved. Legacy
  serialization does not authorize bypassing an old frozen code ledger.

## Actual runner commands

Use a new output root and the same explicit store for shared parents and outputs.
The existing scientific parent selection and observations are unchanged.

    python -m scripts.research.run_query_field_mixture_functions prepare --root NEW_ROOT --store-root STORE
    python -m scripts.research.run_query_field_mixture_functions build --root NEW_ROOT --store-root STORE --workers 1
    python -m scripts.research.run_query_field_confidence_mixture_functions prepare --root NEW_ROOT --store-root STORE
    python -m scripts.research.run_query_field_confidence_mixture_functions build --root NEW_ROOT --store-root STORE --workers 1
    python -m scripts.research.run_query_field_native_modes prepare --root NEW_ROOT --store-root STORE
    python -m scripts.research.run_query_field_native_modes build --root NEW_ROOT --store-root STORE --gpus GPU_IDS

Prediction/score/validation stages use the same --root and --store-root flags.
These examples do not launch or authorize an experiment. Historical reconstruction
uses --legacy-v1 without --store-root and matching scientific input/code ledgers.
Do not reuse an old scientific output root or rewrite its design to accommodate
new code. Compilers refuse existing destinations; partial writes are retained for
inspection and do not become successful construction receipts.

## Readers and resume checks

LERF, NVOS, ScanNet and UQIS public predictors now accept versioned fields. For
LERF/NVOS, keep the scientific config, point its field at the explicit new package,
and pass --store-root STORE. A missing shared store is rejected before encoder
work. The optional replay/export bridge still provides an independent v1 package.

New prediction reports retain v1 archive-SHA semantics and additionally bind
the complete manifest/spec identity. New shared research runs bind caches to
field manifest/state/spec, declared storage root/mode, design/code identity,
query program and current query/config receipts. Native-mode scene skips and
common prediction skips reject missing or mismatched binding receipts.
Common scoring skips additionally bind the prediction barrier, evaluation
configuration, public scorer wrapper and result receipt. This is not recursive
verification of every evaluator dependency or a new fresh hash of every GT,
image or mesh at cache-resume time. Actual predictors retain their original
input validation, and actual field loads hash referenced blobs.

Old designs without the new storage binding preserve historical interpretation.
Existing unbound results are not silently adopted into a new shared-v2 run.
The protected cold validator makes an explicit independent v1 export of a v2
field instead of copying a manifest without its blob closure.

## Tests and review

- Combined targeted CPU suites: 83 passed, 3 CUDA cases deselected, 15.34 seconds,
  job wc_job_hEBPhbCayzCuKYNX. This preceded one additional native-v2-parent case
  and an extra assertion in the same test module; runtime code was unchanged.
- Independent final new-module review: tests/test_query_field_function_storage.py,
  20 passed, 7.52 seconds, job wc_job_iUolm90kn7vTW3OC.
- All three actual runner --help subprocesses passed with root/store/legacy flags.
- Fixtures execute both compilers, source-local function replay, exact v1/v2
  parent state/dtype, different spec identities with zero second-export blob
  bytes, native query parity, float64 preservation, explicit legacy archive SHA,
  a real LERF CPU prediction with deterministic test encoder, child CLI propagation,
  scene-only LERF cohort shape, missing store/closure, tampered manifest,
  prediction cache identity and scoring config changes.
- Independent review found and closed two real wiring defects before completion:
  missing store/root flags on the native scene child, and LERF cohorts using
  scene rather than name. No GPU, real encoder, full benchmark or large payload
  migration was performed. These counts overlap; do not sum them.

## Preservation and outstanding boundaries

Before/after sources and a precise patch are in
experiments/storage_writer_fix_20261013_source_snapshot/.
The before snapshot preserves all 106 files (646,331 bytes) in the previously
validated next20 runtime-code ledger, each matched to its original SHA. Five
additional modified scripts are also preserved with pre-change bytes. The
after snapshot has the 13 changed runtime files and the new CPU test module.

The next20 v2 payloads, mapping, old designs and overlays remain unchanged.
The 20 old payloads plus standalone export are still retained, without deletion
approval. The previous cold parent-chain acceptance was for the previous frozen
code. Its runtime code check intentionally rejects this changed worktree.
Before any later retirement, create a separate explicit new runtime overlay and
repeat the full cold parent-chain verification; never update the old ledger
silently. The preserved source makes the previous exact code recoverable.

Of the original 25 direct legacy writer files, region-semantic,
joint-observation and native-modes are now migrated. The remaining 22 are
unmigrated old writers; current activity has not all been audited. The additional
copy-based mixture/confidence writer is covered here. Initial observation caches,
dense prediction retention, historical five-predict scheduling and the other
writers are not claimed to be migrated.

The separate read-only intermediate audit is
experiments/2026-10-13-intermediate-storage-method-audit.json. Following the
user's explicit five-file confirmation, those interrupted CPU checkpoints were
retired at 07:55 UTC; see experiments/2026-10-13-cpu-trial5-retirement.json. The 62 incomplete
assignment files were subsequently retired after explicit approval to abandon the
attempt at 08:12 UTC; see experiments/2026-10-13-horns-assignments62-retirement.json.
This does not establish bit-identical reproducibility. Native FP32 initial state cannot be exactly reconstructed
from quantized FP16 view caches, and NVOS AP requires continuous score information.
Only the five CPU checkpoints and the subsequently approved 62 assignment files
were included in those two additional retirement batches.
