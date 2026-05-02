# RADIO-GS Framework Figure Prompt

Use this prompt for GPT-image-2, another image model, or manual drawing.

```text
Create a clean academic framework diagram for a CVPR/ICCV-style paper titled "RADIO-GS: Foundation Feature Reconstruction in 3D Gaussian Scenes".

Canvas and style:
- Wide landscape figure, white background, vector-like flat design, crisp readable labels, no decorative gradients, no 3D clipart, no fake quantitative numbers.
- Use consistent color coding: geometry/rendering in blue, foundation features in green, training supervision/losses in orange, evaluation outputs in purple.
- Use thin arrows with clear direction. Use compact modules with rounded corners only for pipeline blocks. Keep text short and publication-ready.

Main layout:
1. Left input block:
   - "Posed RGB frames"
   - "Frozen 3DGS geometry"
   - "Frozen C-RADIOv4-H teacher"
   Show RGB frames and camera frustums feeding into the teacher and Gaussian scene.

2. Training row:
   - Frozen C-RADIOv4-H extracts dense 1280-D teacher features from posed RGB frames.
   - RADIO-GS stores compact scene features with two branches:
     a) "Per-Gaussian latent features"
     b) "Coarse spatial hash branch"
   - Both branches feed into "HCD bottleneck + decoder".
   - The Gaussian renderer produces a "Rendered RADIO-like feature map".
   - A "Screen-space refiner" improves the rendered latent/feature map using rendered geometry cues.
   - Loss arrows compare rendered features against teacher features:
     "feature reconstruction loss"
     "FDH frozen depth-head consistency"
     "regularization / mask-aware training"

3. Inference row:
   - Novel camera view enters "Gaussian feature rendering".
   - Output is "Novel-view foundation feature map".
   - The output branches into downstream frozen or lightweight heads:
     "SigLIP2 text grounding"
     "ScanNet point-query transfer"
     "Depth / segmentation probes"

4. Right evaluation block:
   - Show qualitative LERF heatmap overlays.
   - Show ScanNet point-cloud colored predictions.
   - Show a small runtime/profile icon labelled "profiled evaluation".

Important constraints:
- Do not draw external baselines or invented performance numbers.
- Make RADIO-GS visually central as the 3D feature-memory component.
- Make it clear that the RADIO teacher is frozen and only used for distillation.
- Make it clear that the downstream text/query heads consume rendered features without retraining the 3D scene.
- The final diagram should look like a method overview figure, not a marketing graphic.
```

