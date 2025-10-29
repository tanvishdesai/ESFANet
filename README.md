## MMLLM-MedXAI

End-to-end experimentation for breast cancer histopathology classification with two complementary solution tracks:

- Centralized attention-enhanced CNNs (EFSANet and strong pretrained backbones with attention)
- Federated learning with alert-optimized aggregation (FedAlert), communication efficiency, and optional privacy

This repository contains complete training scripts, ablations, and logged results for both directions on PatchCamelyon and BreakHis histopathology datasets.

### What this project aims to achieve
- Build accurate and efficient classifiers for histopathology patches
- Explore novel attention mechanisms that combine edge-aware spatial and frequency-aware channel cues
- Compare against strong baselines (ResNet50, DenseNet121, VGG16) augmented with attention layers (plain attention and CBAM)
- Design a practical federated learning pipeline tailored for clinical alerting: high precision at an operating threshold, communication-efficient, and privacy-conscious

### Two solution tracks at a glance
- Centralized models (folder: `esfanet/`)
  - EFSANetV2: DenseNet121 backbone → Edge-Frequency Attention → Transformer encoder → classifier
  - Ablations: no transformer; no edge-aware spatial attention; no frequency-aware channel attention
  - Baselines with attention: AttnResNet50, AttnDenseNet121, AttnVGG16, and CBAM-DenseNet
- Federated models (file: `fdl.py`)
  - Transfer learning backbone (MobileNetV2 by default)
  - FedProx, focal loss, adaptive class weighting
  - FedAlert: a novel alert-optimized loss and aggregation weighting targeting clinical thresholds
  - Communication compression (top-k sparsification), optional quantization, optional DP

---

### Repository structure
- `esfanet/`
  - `EFSANetV2.py`: Main centralized model with DenseNet121 + Edge-Frequency Attention + Transformer
  - `EFSANetV2 breakhis.py`: Same model adapted for BreakHis naming/labels
  - `backboene + full attention(no transformer).py`: Ablation (backbone + full attention, no transformer)
  - `full (no Edge-Aware Spatial Attention).py`: Ablation (frequency-only attention)
  - `full ( no Frequency-Aware Channel Attention ).py`: Ablation (edge-only attention)
  - `attnresnet50.py`, `attndensenet.py`, `attnvgg16.py`: Baselines with lightweight attention heads
  - `CBAMdensenet.py`: DenseNet121 with CBAM (channel + spatial attention)
  - `core_idea.md`: Research concept for a sequential agent variant (CNN + SSM + RL) – not implemented here
  - `results.txt`: Aggregated centralized model results (PatchCamelyon + BreakHis)
- `fdl.py`: Full federated learning pipeline with FedAlert, FedProx, TL backbones, compression, and (optional) DP
- `model-1-results.txt`..`model-4-results.txt`: Federated experiments logs for variants (FedAvg baseline, FedProx, with/without FedAlert)

---

## 1) Centralized models (esfanet)

### Datasets
- PatchCamelyon: PNG files or Kaggle standard naming with `_class0`/`_class1` in filename used by several scripts
- BreakHis: uses directory and filename conventions to derive benign/malignant labels in `EFSANetV2 breakhis.py`

Expected defaults in the scripts:
- Image size: 96×96
- Normalization: ImageNet stats
- Loss: FocalLoss (class imbalance), sometimes class weighting
- Optimizers: Adam/AdamW; schedulers: CosineAnnealingLR or ReduceLROnPlateau

### Key model: EFSANetV2
Pipeline:
1. DenseNet121 (pretrained, features only)
2. Edge-Frequency Attention
   - Edge-aware spatial attention using fixed Sobel filters per channel
   - Frequency-aware channel attention via FFT magnitude → squeeze → MLP → sigmoid
   - Combined multiplicative attention with residual connection
3. Transformer encoder block (multi-head self-attention, LN, FFN)
4. Token pooling (mean) → classifier head

Variants and ablations:
- No Transformer: `backboene + full attention(no transformer).py`
- Edge-only: `full ( no Frequency-Aware Channel Attention ).py`
- Frequency-only: `full (no Edge-Aware Spatial Attention).py`
- Baselines with attention: `attnresnet50.py`, `attndensenet.py`, `attnvgg16.py`, `CBAMdensenet.py`

### How to run (centralized)
Run any script directly with Python. Update `DATA_DIR` paths at the top of each file to point to your dataset.

Example (PatchCamelyon-style data):
```bash
python esfanet/EFSANetV2.py
```

Example (BreakHis directory):
```bash
python "esfanet/EFSANetV2 breakhis.py"
```

Example (ablations and baselines):
```bash
python "esfanet/backboene + full attention(no transformer).py"
python "esfanet/full (no Edge-Aware Spatial Attention).py"
python "esfanet/full ( no Frequency-Aware Channel Attention ).py"
python esfanet/attnresnet50.py
python esfanet/attndensenet.py
python esfanet/attnvgg16.py
python esfanet/CBAMdensenet.py
```

---

## 2) Federated learning (fdl.py)

### Overview
The federated pipeline simulates multiple clients (e.g., hospitals) with non-IID data using PatchCamelyon parquet files downloaded from Hugging Face. It emphasizes:
- Alert-centric learning: optimize the model specifically at a clinically relevant operating threshold via a new FedAlert loss and alert-quality-weighted aggregation
- Communication efficiency: top-k sparsification of model updates; optional quantization
- Stability on non-IID data: FedProx proximal regularization
- Class imbalance robustness: focal loss and aggressive class weighting
- Optional privacy via differential privacy noise and clipping

### Components
- Transfer learning backbones: MobileNetV2 (default), EfficientNet-B0 (via `timm`, optional)
- FedAlert loss: combines base loss with differentiable penalties for false alarms and missed detections at a threshold (default 0.75)
- FedAlert aggregation: re-weights clients by F1 at the alert threshold multiplied by data size
- Communication: top-k sparsification, decompression on server, dtype safety fixes
- Optional DP: Gaussian noise on floating tensors, norm clipping, skip BN buffers

### How to run (federated)
```bash
python fdl.py
```

What it does:
- Downloads PatchCamelyon from Hugging Face into local parquet shards
- Creates non-IID client splits with a Dirichlet partition
- Trains for multiple rounds with local epochs per client, aggregates on server
- Saves metrics, communication stats, alert logs, and plots

Key toggles in `Config` (inside `fdl.py`):
- `USE_FEDALERT`, `FEDALERT_ALPHA`, `FEDALERT_BETA`, `ALERT_THRESHOLD`
- `USE_FEDPROX`, `FEDPROX_MU`
- `USE_TRANSFER_LEARNING`, `PRETRAINED_MODEL`, `FREEZE_LAYERS`, `UNFREEZE_AFTER_ROUNDS`
- `COMPRESSION_RATIO`, `QUANTIZATION_BITS`
- `USE_DIFFERENTIAL_PRIVACY`, `DP_NOISE_MULTIPLIER`, `DP_CLIP_NORM`

Outputs:
- `training_metrics.csv`, `communication_stats.csv`, `alert_logs.csv`
- `federated_training_metrics.png`, `confusion_matrix.png`, `roc_curve.png`
- `final_global_model.pth`

---

## Environment and setup

### Python dependencies
- torch, torchvision, torchaudio (per your CUDA setup)
- numpy, pandas, scikit-learn, tqdm, matplotlib, seaborn, pillow
- datasets (Hugging Face) for `fdl.py`
- timm (optional, for EfficientNet-B0 in `fdl.py`)

Example install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tqdm matplotlib seaborn pillow datasets
# Optional
pip install timm
```

GPU is recommended for all training.

---

## Results summary

Centralized (from `esfanet/results.txt`, PatchCamelyon unless noted):
- AttnDenseNet: Acc 0.9639, Prec 0.9209, Rec 0.9548, F1 0.9375
- CBAM DenseNet: Acc 0.9647, Prec 0.9254, Rec 0.9525, F1 0.9388
- AttnResNet50: Acc 0.9441, Prec 0.8784, Rec 0.9320, F1 0.9044
- AttnVGG16: Acc 0.9439, Prec 0.8731, Rec 0.9390, F1 0.9049
- EFSANet (proposed): Acc 0.9638, Prec 0.9562, Rec 0.9547, F1 0.9555
- EFSANet (BreakHis): Acc 0.9608, Prec 0.9559, Rec 0.9528, F1 0.9543
- Ablation (no spatial): Acc 0.9644, Prec 0.9576, Rec 0.9547, F1 0.9561
- Ablation (attention only, no transformer): Acc 0.9661, Prec 0.9589, Rec 0.9577, F1 0.9583
- Ablation (no frequency): Acc 0.9639, Prec 0.9559, Rec 0.9553, F1 0.9556

Takeaways:
- EFSANet and its ablations consistently perform at ~96–97% accuracy with balanced macro precision/recall
- Removing either spatial or frequency pieces yields very similar performance; attention-only without transformer is slightly best here
- Attention-augmented DenseNet outperforms ResNet50/VGG16 variants

Federated (from `model-*.txt`, PatchCamelyon; non-IID, MobileNetV2 TL):
- FedAvg baseline (Model 1): Acc 0.8226, F1 0.8059, AUC 0.9063
- FedProx (Model 2): Acc 0.7348, F1 0.6758, AUC 0.8684
- Your method with FedAlert aggregation (Model 3): Acc 0.8185, F1 0.8124, AUC 0.9061
- Your method without FedAlert (consensus aggregation, Model 4): Acc 0.7736, F1 0.7947, AUC 0.8310

Takeaways:
- Alert-aware training and aggregation (FedAlert) maintain strong AUC (~0.91) and balanced F1 in non-IID settings
- Communication compression and transfer learning enable practical end-to-end rounds

---

## Notes and limitations
- File names with spaces in `esfanet/` are preserved; run them by quoting the path
- Centralized scripts assume local PNG datasets and may need `DATA_DIR` edits
- Federated script downloads parquet via Hugging Face and shards locally; ensure internet access and disk space
- `core_idea.md` outlines a research direction (SSM + RL agent) not implemented in code here

---

## Citation
If you use this repository, please cite this project and the relevant datasets (PatchCamelyon, BreakHis) and backbones (DenseNet, ResNet, VGG, MobileNet) accordingly.


