# ScanNet-UQIS LUDVIG image-adapter smoke — 2026-08-13

## Authority

This is an exact-runtime integration smoke, not a ScanNet-UQIS metric row.
The fixture uses a real legacy PFIR crop and the official ScanNet mesh for
`scene0050_02`, but it predates the final UQIS target and frame-exclusion
receipts. Consequently:

- `result_eligible=false`;
- `formal_benchmark_row_eligible=false`;
- `official_ludvig_reproduction=false`;
- `paper_metric_comparable=false`;
- evaluator metrics were not opened.

The fixture and run manifests are the authority for all values below.

## Exact path exercised

The adapter executed:

1. the vendored LUDVIG DINOv2 ViT-G/14 query encoder on one re-encoded
   224×224 RGB crop;
2. the frozen scene-specific Phase-B standardization/PCA40 transform;
3. center 3×3 pooling and L2 normalization;
4. cosine response against the frozen 300,000×40 Phase-C Gaussian field;
5. log-stable, opacity-weighted Gaussian readout at every official ScanNet
   mesh vertex using K=64 and `pinv(covariance + 1e-6 I)`;
6. the frozen monotonic sigmoid with scale 1 and bias 0.

The adapter process received and opened no evaluator-private target identity,
instance label, paired query, depth, or mask. The separate smoke-fixture
constructor parsed the official ScanNet aggregation and segmentation only to
export and verify official mesh order; it discarded instance IDs and emitted
no evaluator manifest.

## Bound inputs

- Phase-B manifest SHA-256:
  `1d546a335e2f3ec807c69b23f06a7876d1a325d53e16a94b0d64f8b7556d147b`
- Phase-C manifest SHA-256:
  `dcf8d864da50aa455f805d94bce707daf4cfc4ac1f602ea03141b38a57eb13fb`
- Gaussian feature SHA-256:
  `979725de417e301991199f62bffee520d82cd38fb25bb119dd470d8ae0134e5f`
- official mesh XYZ SHA-256:
  `81af14954fc7ff32a19cf172dc62e000d335cdfa502cafedb002c4f25bfc93a2`
- smoke query-manifest SHA-256:
  `632ef10c5a36d1aa50535cd24543b1258cf42c431866b86747c6d49c9430d895`

## Observed output

- query: `uq_f8eec7c538fc6bfb114020829f8ab968`
- output: finite `float32[196815]` fixed-sigmoid scores in the official-mesh
  probability domain
- probability range: `[0.3397446275, 0.6991129518]`
- probability SHA-256:
  `ebd850bc130c068f3258f57d8b012736aa286c9a3629c76069c792ede149ad8c`
- descriptor shape: `[1, 40]`
- elapsed wall time: `30.20295627 s`
- peak CUDA allocated/reserved: `6,904,893,440 / 7,077,888,000 bytes`
- run-manifest SHA-256:
  `1ce52713e75dcfab75dfb75fc6ecab1b4bb0c8d55d08bc830a38ffcb1c57c4fb`

The normalized K=64 readout produced a value for every vertex, but this is not
a field-coverage claim: 163 vertices had unclipped kernel mass below the
smallest positive `float32` value and are conditional-kernel extrapolations.

## Reproduction command

```bash
LD_LIBRARY_PATH=/root/baselines/LUDVIG/.driver535 \
PYTHONPATH=/root/RADIO-GS:/root/baselines/LUDVIG/.reproduction-deps-sm86 \
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/run_uqis_image.py \
  --query-manifest \
    /mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_10_v0_1/ludvig_scene0050_02_exact_runtime_smoke_fixture_v1/query_manifest.image.json \
  --phase-b-dir \
    /root/ludvig_pfpr_runs/pfpr_scene0050_02_exact_ludvig_vendored_dino_pca_full120_v2_driver535 \
  --phase-b-manifest-sha256 \
    1d546a335e2f3ec807c69b23f06a7876d1a325d53e16a94b0d64f8b7556d147b \
  --phase-c-dir \
    /root/ludvig_pfpr_runs/pfpr_scene0050_02_exact_ludvig_phase_c_full120_v1_driver535 \
  --phase-c-manifest-sha256 \
    dcf8d864da50aa455f805d94bce707daf4cfc4ac1f602ea03141b38a57eb13fb \
  --output-dir /a/new/output/directory
```

The adapter implementation refuses to overwrite an existing output directory.
The retained completed run is under
`/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_10_v0_1/ludvig_scene0050_02_exact_runtime_smoke_run_v3`.
