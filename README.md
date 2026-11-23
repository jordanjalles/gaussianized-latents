# Gaussianized latents

Shaping an autoencoder's latent into a bell curve so a generator can learn it, then using that as anti-collapse for masked-view (JEPA-style) alignment.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `ae_model.py`: the autoencoder module
- `STL10-AE-wasserstein-2.ipynb`: sliced, token-wise, channel-wise and batch-wise Wasserstein losses
- `STL10-AE-sigreg-langrange.ipynb`, `SIGReg.ipynb`: SIGReg (from LeJEPA) with a Lagrange multiplier
- `STL10-AE-wass-masked-gram.ipynb`, `STL10-AE-jepa-recon.ipynb`: masked views aligned to a stop-gradient clean view

## Running it

**Data:** set `ML_DATASETS` to the folder that holds the datasets (torchvision downloads STL-10 / MNIST there). Paths like `<home>` and `<data-root>` are placeholders for my machines.

## Results

Cosine similarity between latents became bell-shaped and the PCA spectrum flattened. Sliced Wasserstein alone lets structure hide along the token and channel axes, so the per-axis versions matter. Three bugs slowed this phase down: missing decoder slot embeddings, a misplaced parenthesis in the RoPE frequencies, and projecting the latent on the token axis.

**Code:** written by hand in Jupyter notebooks, with a chatbot as a tutor for questions and plotting helpers.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
