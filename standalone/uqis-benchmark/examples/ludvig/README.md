# LUDVIG adapter example

`text_diffusion.py` is the dependency-light benchmark-local graph operator used
in the reported text comparator. It accepts an already aligned CLIP relevance
vector, DINO features and Gaussian XYZ; it does not load RGB, SAM or evaluator
labels.

The complete field construction adapter is not copied into the installable
benchmark core because it binds a specific upstream LUDVIG/3DGS checkout,
patched rasterizer, checkpoints and CUDA environment. Keeping method-specific
training code outside the Evaluation Authority module is intentional.

To reproduce the reported comparator, follow
[`docs/ludvig-reproduction.md`](../../docs/ludvig-reproduction.md) and provide:

- a content-bound OpenCLIP field on the full Gaussian carrier;
- a content-bound pruned DINO/PCA40 field;
- the frozen full-to-pruned `source_indices.npy`;
- official-mesh XYZ and a continuous Gaussian readout implementation;
- one-query read-only workspaces from `uqis-stage-workspace`.

Before publication, export the RADIO-GS LUDVIG adapters into a separate
optional package only after upstream source/checkpoint licenses and environment
locks have been reviewed.
