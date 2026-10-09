# Extending Reconstruction: Reconstruction-to-Vector Diffusion for Hyperspectral Anomaly Detection

[IEEE Transactions on Geoscience and Remote Sensing](https://ieeexplore.ieee.org/document/11720431)

## Abstract

While Hyperspectral Anomaly Detection (HAD) excels at identifying sparse targets in complex scenes, conventional reconstruction-based models typically rely on a scalar metric as the final detection criterion. This heavy reliance on magnitude-only scalar residuals frequently triggers sub-pixel anomaly vanishing during spatial downsampling, alongside confirmation bias when unpurified targets corrupt background representations. In this paper, we propose Reconstruction-to-Vector Diffusion (R2VD), which extends the reconstruction paradigm by repurposing it for background manifold purification, followed by high-dimensional generative score modeling. Our framework introduces a four-stage pipeline: (1) a Physical Prior Extraction (PPE) stage that mitigates early confirmation bias via dual-stream statistical guidance; (2) a Guided Manifold Purification (GMP) stage utilizing an OmniContext Autoencoder (OCA) to extract purified residual maps while preserving fragile sub-pixel topologies; (3) a Residual Score Modeling (RSM) stage where a Diffusion Transformer (DiT), guarded by a Physical Spectral Firewall (PSF), effectively isolates cross-spectral leakage; and (4) a Vector Dynamics Inference (VDI) stage that robustly decouples targets from backgrounds by evaluating high-dimensional vector interference patterns instead of conventional scalar errors. Comprehensive evaluations on eight datasets confirm that R2VD establishes a new state-of-the-art, delivering exceptional target detectability and background suppression. The code is available at https://github.com/Bondojijun/R2VD.

## Requirements

Python 3.12 and the packages in `requirements.txt`. For an NVIDIA GPU with CUDA 12.8 support, install the matching PyTorch wheel before the remaining dependencies:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

PyTorch also supports CPU execution; install the appropriate PyTorch build for your system before installing `requirements.txt`.

## Run

The default input is `datasets/ABU-Urban-2.mat`. To select another dataset, change `MAT_FILE` near the top of `main.py` to the corresponding file in `datasets/`. The MATLAB file must contain the `data` and `map` keys.

```bash
python main.py
```

The console reports the dataset name, DiT training epochs, inference progress, and final AUC. Results are saved to `results/<dataset-name>/anomaly_detection.png`. The first run also saves `results/<dataset-name>/pipeline_cache.npz`; subsequent runs reuse this cache to skip pipelines 1 and 2. DiT training and inference still run each time.
