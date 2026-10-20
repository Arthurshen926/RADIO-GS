"""Bounded CPU inference equivalence probes for retained RADIO-GS state.

This module reads no checkpoints, configs, datasets, PLYs, or teacher weights.
The caller supplies a trusted, CPU-loaded checkpoint and resolved config.
It runs real repository modules on 16 deterministic selected primitive rows
and synthetic 4x4 maps. It does NOT validate rendering, KNN, scene bounds,
camera calibration, full-scene metrics, or benchmark accuracy.

Large retained tensors are assigned by reference, never copied or reduced.
Use mmap checkpoint loading in the caller if avoiding a full storage read.
Uses meta-device construction and strict by-reference state installation on PyTorch 2.0+.
No training, optimizer creation, gradients, GPU, downloads, or file writes.
"""

from collections.abc import Mapping
import math


def probes(ckpt, cfg, *, loaded_modules=None):
    """Return dict[str, CPU Tensor] with deterministic named probe outputs.

    A strict state-dict mismatch raises; there is no partial warmstart or
    permissive fallback. Optional modules are exercised when their state key
    exists. Unsupported extra state dictionaries raise rather than being
    silently described as covered.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
    from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
    from radio_gs.models.hcd_codec import build_feature_codec
    from radio_gs.models.point_summary_adapter import (
        CompactToSummaryAdapter,
        append_point_summary_context,
        point_summary_context_dim,
    )
    from radio_gs.models.featsharp_3d import FeatSharp3D
    from radio_gs.models.screen_refiner import (
        ScreenSpaceRefiner,
        build_refiner_guide,
        compute_refiner_extra_channels,
    )
    from radio_gs.heads.depth_head import DepthHead
    from radio_gs.heads.segmentation_head import SegmentationHead

    get = cfg.get if isinstance(cfg, Mapping) else lambda k, d=None: getattr(cfg, k, d)
    supported = {
        "model_state_dict", "codec_state_dict", "sharpener_state_dict",
        "refiner_state_dict", "depth_head_state_dict", "seg_head_state_dict",
        "point_summary_adapter_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "scaler_state_dict",
    }
    unknown = sorted(k for k in ckpt if k.endswith("_state_dict") and k not in supported)
    if unknown:
        raise ValueError("Uncovered module state dictionaries: " + repr(unknown))

    def load(module, key):
        state = ckpt[key]
        supplied = (loaded_modules or {}).get(key)
        if supplied is not None:
            actual = supplied.state_dict()
            if set(actual) != set(state) or any(
                actual[k].dtype != v.dtype or actual[k].shape != v.shape
                or not torch.equal(actual[k].detach().cpu(), v)
                for k, v in state.items()
            ):
                raise ValueError("Public builder state differs: " + key)
            return supplied.eval()
        for name, tensor in state.items():
            if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                raise ValueError(f"{key}.{name} must already be on CPU")
        current = module.state_dict()
        if set(current) != set(state):
            raise ValueError(f"Strict {key} keys mismatch: missing={set(current)-set(state)}, extra={set(state)-set(current)}")
        for name, tensor in state.items():
            if current[name].shape != tensor.shape or current[name].dtype != tensor.dtype:
                raise ValueError(f"Strict {key}.{name} shape/dtype mismatch")
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            if leaf in parent._parameters:
                parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            elif leaf in parent._buffers:
                parent._buffers[leaf] = tensor
            else:
                raise ValueError(f"Unregistered state: {key}.{name}")
        # Actual repository module loader runs strictly; every target already
        # references the exact CPU tensor, so this is not a second large copy.
        module.load_state_dict(state, strict=True)
        module.eval()
        return module

    out = {}

    def record(name, value):
        if isinstance(value, Mapping):
            for key, tensor in sorted(value.items()):
                record(name + "/" + str(key), tensor)
            return
        if not torch.is_tensor(value):
            raise TypeError(f"Non-tensor probe result {name}: {type(value)}")
        if value.device.type != "cpu":
            raise ValueError(f"Probe {name} escaped CPU")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite probe output: {name}")
        out[name] = value.detach().clone()

    def wave(shape, offset=0.0):
        n = math.prod(shape)
        return torch.sin(torch.arange(n, dtype=torch.float32) * 0.017 + offset).reshape(shape)

    arch = get("architecture", "explicit")
    state = ckpt["model_state_dict"]
    with torch.device("meta"):
        if arch == "hybrid":
            model = HybridFeatureGaussian(
                latent_dim=get("hybrid_latent_dim", 16),
                hash_output_dim=get("hash_output_dim", 48),
                fine_dim=get("fine_dim", 64),
                coarse_dim=get("coarse_dim", 64),
                output_dim=get("hybrid_output_dim", 128),
                num_levels=get("hash_levels", 16),
                features_per_level=get("hash_features_per_level", 2),
                log2_hashmap_size=get("hash_log2_size", 19),
                base_resolution=get("hash_base_resolution", 16),
                max_resolution=get("hash_max_resolution", 2048),
                decoupled_heads=get("hybrid_decoupled_heads", False),
                use_semantic_adaptor=get("hybrid_semantic_adaptor", False),
                semantic_adaptor_mode=get("hybrid_semantic_adaptor_mode", "confidence"),
                semantic_adaptor_hidden_dim=get("hybrid_semantic_adaptor_hidden_dim", 64),
                semantic_adaptor_use_geometry_guidance=get("hybrid_semantic_adaptor_use_geometry_guidance", True),
                semantic_adaptor_use_depth_guidance=get("hybrid_semantic_adaptor_use_depth_guidance", False),
                semantic_adaptor_residual=get("hybrid_semantic_adaptor_residual", True),
                use_quality_head=get("hybrid_quality_head", False),
                use_visibility_head=get("hybrid_visibility_head", False),
            )
            feature_name = "_latent"
            latent_dim = get("hybrid_latent_dim", 16)
        elif arch == "explicit":
            model = ExplicitFeatureGaussian(
                latent_dim=get("latent_dim", 64), train_sh=get("train_sh", False)
            )
            feature_name = "_feature"
            latent_dim = get("latent_dim", 64)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        # Geometry/primitive tensors normally acquire their shape from PLY.
        # Only shape and dtype metadata are used here; strict assignment then
        # retains the caller's complete original CPU arrays without a copy.
        for key in ("_xyz", "_rotation", "_scaling", "_opacity", "_features_dc", "_features_rest"):
            if key in state:
                model._buffers[key] = torch.empty_like(state[key], device="meta")
        setattr(model, feature_name, nn.Parameter(torch.empty_like(state[feature_name], device="meta")))
        for key in ("_sh_dc_param", "_sh_rest_param"):
            if key in state:
                setattr(model, key, nn.Parameter(torch.empty_like(state[key], device="meta")))
        codec = build_feature_codec(
            input_dim=get("radio_feature_dim", 1280),
            bottleneck_dim=get("bottleneck_dim", 64),
            codec_type=get("codec_type", "hcd"),
            dual_stream=get("dual_stream", True),
            symmetric_decoder=get("symmetric_decoder", False),
            hidden_normalization=get("codec_hidden_normalization", "legacy_group"),
            final_normalization=get("codec_final_normalization", "legacy_group"),
        )
    model = load(model, "model_state_dict")
    codec = load(codec, "codec_state_dict")

    # Bound compute and leave process CPU settings as found.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 2))
    try:
        with torch.inference_mode():
            rows = state[feature_name].shape[0]
            if rows < 1:
                raise ValueError("Checkpoint has no primitive feature rows")
            ids = torch.linspace(0, rows - 1, 16, dtype=torch.float64).round().long()
            features = getattr(model, feature_name)[ids].float()
            if arch == "explicit":
                # Algebraically identical to get_features()[ids], without
                # normalizing every row in the geometry cloud.
                features = F.normalize(features, p=2, dim=-1)
            record("sample/row_indices", ids)
            record("sample/latent_rows", features)
            latent_map = features.T.reshape(1, latent_dim, 4, 4)
            points = torch.tensor([
                [0.02, 0.13, 0.29], [0.97, 0.83, 0.71],
                [0.19, 0.67, 0.43], [0.81, 0.37, 0.59],
            ], dtype=torch.float32).repeat(4, 1)
            position_map = points.T.reshape(1, 3, 4, 4)
            depth_map = torch.linspace(0.5, 2.0, 16).reshape(1, 1, 4, 4)

            def core(prefix, latents):
                if arch == "hybrid":
                    result = model.decode_screen_space(
                        latents, position_map, return_aux=True, depth_map=depth_map
                    )
                    record(prefix, result)
                    return result["features"] if isinstance(result, Mapping) else result
                record(prefix, latents)
                return latents

            compact_map = core("core/synthetic_screen", latent_map)
            compact_rows = compact_map.permute(0, 2, 3, 1).reshape(16, -1)
            if arch == "hybrid" and not (
                get("hybrid_semantic_adaptor", False)
                and get("hybrid_semantic_adaptor_use_depth_guidance", False)
            ):
                record("core/point_decoder", model._decode_point_features(
                    features, points, return_aux=True
                ))
            decoded = codec.decode(compact_map.float())
            record("codec/decode_screen", decoded)
            record("codec/decode_points", codec.decode_points(compact_rows.float()))
            encoded = codec.encode(wave((1, get("radio_feature_dim", 1280), 4, 4), 0.21))
            record("codec/encode_synthetic", encoded)
            record("codec/reconstruct_synthetic", codec.decode(encoded))

            if "sharpener_state_dict" in ckpt:
                with torch.device("meta"):
                    sharpener = FeatSharp3D(
                        mode=get("featsharp_mode", "analytical"),
                        feature_dim=latent_dim,
                        strength=get("featsharp_strength", 0.5),
                    )
                sharpener = load(sharpener, "sharpener_state_dict")
                kwargs = {}
                if sharpener.mode == "multiview":
                    eye = torch.eye(4).unsqueeze(0)
                    kwargs = dict(
                        features_sources=[latent_map.flip(-1), latent_map * 0.9],
                        depths_ref=depth_map[:, 0], viewmats_ref=eye,
                        viewmats_sources=[eye, eye], K=torch.eye(3),
                    )
                record("sharpener/synthetic_map", sharpener(latent_map, **kwargs))

            if "refiner_state_dict" in ckpt:
                guide_flags = dict(
                    rgb_guide=get("refiner_rgb_guide", False),
                    depth_guide=get("refiner_depth_guide", False),
                    depth_grad=get("refiner_depth_grad", False),
                    alpha_guide=get("refiner_alpha_guide", False),
                    boundary_guide=get("refiner_boundary_guide", False),
                )
                extra = compute_refiner_extra_channels(**guide_flags)
                with torch.device("meta"):
                    refiner = ScreenSpaceRefiner(
                        latent_dim=latent_dim,
                        hidden_dim=get("refiner_hidden_dim", 128),
                        num_blocks=get("refiner_num_blocks", 4),
                        dropout=get("refiner_dropout", 0.1),
                        extra_channels=extra,
                        norm_type=get("refiner_norm_type", "gn"),
                    )
                refiner = load(refiner, "refiner_state_dict")
                guide = build_refiner_guide(
                    {"depth_map": depth_map, "alpha_map": wave((1, 1, 4, 4), 0.4).sigmoid()},
                    rgb_guide=wave((1, 3, 4, 4), 0.8) if guide_flags["rgb_guide"] else None,
                    use_depth_guide=guide_flags["depth_guide"],
                    use_depth_grad=guide_flags["depth_grad"],
                    depth_grad_scale=get("refiner_depth_grad_scale", 10.0),
                    use_alpha_guide=guide_flags["alpha_guide"],
                    use_boundary_guide=guide_flags["boundary_guide"],
                )
                refined = refiner(latent_map, guide=guide)
                record("refiner/synthetic_map", refined)
                refined_compact = core("refiner/core_synthetic_screen", refined)
                record("refiner/codec_decode_screen", codec.decode(refined_compact.float()))

            if "depth_head_state_dict" in ckpt:
                with torch.device("meta"):
                    head = DepthHead(
                        feature_dim=get("radio_feature_dim", 1280),
                        hidden_dim=get("depth_head_hidden_dim", 256),
                        num_layers=get("depth_head_num_layers", 3),
                        head_type=get("depth_head_type", "mlp"),
                    )
                head = load(head, "depth_head_state_dict")
                record("depth_head/prediction", head(decoded.float()))

            if "seg_head_state_dict" in ckpt:
                with torch.device("meta"):
                    head = SegmentationHead(
                        feature_dim=get("radio_feature_dim", 1280),
                        num_classes=get("seg_num_classes", 40),
                        hidden_dim=get("seg_head_hidden_dim", 256),
                        num_layers=get("seg_head_num_layers", 2),
                        head_type=get("seg_head_type", "mlp"),
                    )
                head = load(head, "seg_head_state_dict")
                logits = head(decoded.float())
                record("seg_head/logits", logits)
                record("seg_head/argmax", logits.argmax(dim=1))

            if "point_summary_adapter_state_dict" in ckpt:
                context = str(get("point_summary_adapter_context_features", "") or "")
                input_dim = get("bottleneck_dim", get("hybrid_output_dim", 128)) + point_summary_context_dim(context)
                with torch.device("meta"):
                    adapter = CompactToSummaryAdapter(
                        input_dim=input_dim, output_dim=1536,
                        hidden_dim=get("point_summary_adapter_hidden_dim", 512),
                        num_layers=get("point_summary_adapter_num_layers", 2),
                        dropout=get("point_summary_adapter_dropout", 0.0),
                    )
                adapter = load(adapter, "point_summary_adapter_state_dict")
                adapter_input = append_point_summary_context(
                    compact_rows, context_features=context,
                    opacity=torch.sigmoid(state["_opacity"][ids].float()),
                    scales=torch.exp(state["_scaling"][ids].float()),
                    view_counts=torch.arange(1, 17, dtype=torch.float32),
                    view_count_max=16.0,
                )
                summary = adapter(adapter_input)
                visual = F.normalize(summary.float(), dim=-1)
                # Fixed synthetic query embeddings test the same normalized
                # visual @ text.T operation used by the official evaluator.
                # These are NOT actual class embeddings or benchmark scores.
                text = F.normalize(wave((7, summary.shape[1]), 0.37), dim=-1)
                scores = visual @ text.T
                record("adapter/raw_summary", summary)
                record("adapter/normalized_summary", visual)
                record("adapter/synthetic_query_scores", scores)
                record("adapter/synthetic_query_argmax", scores.argmax(dim=-1))

            return out
    finally:
        torch.set_num_threads(old_threads)
