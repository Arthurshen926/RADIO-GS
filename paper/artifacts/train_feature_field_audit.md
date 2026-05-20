# Train Feature Field Audit

- Script: `radio_gs/scripts/train_feature_field.py`
- Line count: `3735`
- Overall status: `pass`

## Static Checks

| Check | Status | Severity | Evidence | Recommendation |
|---|---|---|---|---|
| script_size | pass | medium | 3735 lines; threshold 4000 | Split training data, losses, trainer loop, and checkpointing into importable modules before release. |
| run_manifest | pass | high | present | Keep run_manifest, git metadata, artifact paths, experiment report, and metrics history in every training run. |
| split_resolution | pass | high | train/val split tokens present | Keep explicit train/val feature, pose, and frame-id resolution to prevent view/feature leakage. |
| trusted_checkpoint_io | pass | high | load_trusted_checkpoint referenced | Load model checkpoints through the trusted checkpoint helper; keep raw torch.load limited to feature/text/cache tensors. |
| training_lock | pass | medium | lock acquisition/release present | Keep per-output training lock to prevent concurrent writers from corrupting run artifacts. |
| raw_tensor_load_sites | pass | medium | 1 raw torch.load site behind or replaced by load_training_tensor_cache | Document each raw torch.load as feature/text/cache tensor loading or move it behind typed loader helpers. |

## Test Coverage Signals

| Coverage target | Status | Evidence | Matched files |
|---|---|---|---|
| frame_order_and_direct_point_visibility | pass | 4 matching test files | tests/test_direct_point_supervision.py, tests/test_generate_scannet_dino_cv_configs.py, tests/test_scannet_og_config_generator.py, tests/test_scannet_teacher_eval.py |
| split_config_generation | pass | 1 matching test files | tests/test_scannet_og_config_generator.py |
| provenance_freeze | pass | 3 matching test files | tests/test_build_submission_freeze_report.py, tests/test_paper_artifact_registry.py, tests/test_verify_submission_provenance.py |
| checkpoint_io | pass | 15 matching test files | tests/test_build_controlled_evidence_table.py, tests/test_build_sam3_foundation_cache.py, tests/test_build_storage_footprint_report.py, tests/test_build_submission_freeze_report.py, tests/test_build_train_feature_field_audit.py, tests/test_checkpoint_io.py, tests/test_generate_scannet_dino_cv_configs.py, tests/test_lerf_direct_3d_selection.py, tests/test_lerf_prompt_sweep.py, tests/test_point_summary_adapter_training.py, tests/test_radio_adaptors.py, tests/test_scannet_og_config_generator.py, tests/test_scannet_v67_queue_launcher.py, tests/test_siglip2_text_encoder.py, tests/test_verify_submission_provenance.py |

## Open Items

- none
