# RADIO-GS Framework Figure Prompt

Use this prompt for GPT-image-2, another image model, or manual drawing.

```text
Create a clean academic framework diagram for a TPAMI-style paper titled "GaussFM: Compact Foundation-Feature Gaussian Memory for Open-Vocabulary 3D Scene Understanding".

Canvas and style:
- Wide landscape figure, white background, vector-like flat design, crisp readable labels, no decorative gradients, no 3D clipart, no fake quantitative numbers.
- Use consistent color coding: geometry/rendering in blue, RADIO feature reconstruction in green, training supervision/losses in orange, query outputs in purple.
- Use thin arrows with clear direction. Use compact modules with rounded corners only for pipeline blocks. Keep text short and publication-ready.

Main layout:
1. Left input block:
   - "Posed RGB frames"
   - "Pretrained 3DGS geometry/RGB"
   - "Frozen C-RADIOv4-H"
   Show RGB frames and camera frustums feeding into RADIO feature extraction and the Gaussian scene.

2. Training row:
   - Frozen C-RADIOv4-H extracts dense 1280-D RADIO reference features from posed RGB frames.
   - GaussFM stores a compact RADIO Gaussian memory with:
     a) "Per-Gaussian compact latent"
     b) "Spatial context branch"
     c) "visibility / confidence cues"
   - These feed into "HCD/CTR bottleneck + RADIO decoder".
   - The Gaussian renderer produces a "Rendered RADIO-like feature map".
   - A "View-conditioned calibration" module improves the rendered latent/feature map using rendered geometry cues.
   - Loss arrows compare rendered features against RADIO reference features:
     "dense RADIO reconstruction"
     "FDH frozen depth-head consistency"
     "DINO/SAM adaptor-space structural regularization"
   - A separate sparse arrow:
     "Multiview Primitive Registration"
     "SigLIP summary semantic anchor"
     "compress registered evidence into compact memory"

3. Query row:
   - Novel camera view enters "Gaussian feature rendering" and outputs "2D RADIO-compatible feature map".
   - Gaussian/point query enters "Decode compact primitive / point feature".
   - Branches:
     "LERF 2D OVS"
     "LERF direct 3D OVS"
     "ScanNet point query"
     "Frozen-head feature-usability probes"

4. Right evaluation block:
   - Show qualitative LERF heatmap overlays.
   - Show ScanNet point-cloud colored predictions.
   - Show a small runtime/profile icon labelled "profiled evaluation".

Important constraints:
- Do not draw external baselines or invented performance numbers.
- Make RADIO-GS visually central as the 3D feature-memory component.
- Make it clear that RADIO is the only raw foundation feature being reconstructed.
- Do not draw DINO, SigLIP2, or SAM3 as parallel input foundation features; draw them only as adaptor-space constraints/probes downstream of RADIO-compatible reconstruction.
- Make it clear that the compact memory is learned on top of pretrained 3DGS geometry; RGB reconstruction is not the contribution.
- Make it clear that downstream text/query heads consume reconstructed features without retraining the 3D scene.
- The final diagram should look like a method overview figure, not a marketing graphic.
```
