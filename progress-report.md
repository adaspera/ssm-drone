# Research Progress Report: SSM-Based Architectures for YOLO Object Detection

## 1. Research Overview

This project investigates integrating **State Space Model (SSM)** architectures — specifically Mamba variants and VMamba cross-scan — into the YOLO26 object detection framework. The goal is to compare hybrid CNN–SSM detectors against standard CNN-only (YOLO26n/s), Transformer-based (RT-DETR, RF-DETR), and dedicated SSM (Wang2024 Mamba-YOLO) baselines on both a general-purpose benchmark (COCO2017) and small-scale drone/bird detection datasets.

---

## 2. Datasets

| Dataset | Classes | Train | Val/Test | Format | Purpose |
|---------|---------|-------|----------|--------|---------|
| **COCO2017** | 80 | 118,287 | 5,000 | YOLO | General-purpose benchmark |
| **mergedv2** | 2 (drone, bird) | 3,265 | 716 | YOLO | Primary task-specific evaluation |
| **mergedv3** | 1 (drone) | 13,962 | 3,393 | YOLO | Larger drone-only dataset |
| **UAV-roboflow** | 1 (drone) | 7,503 | 237 | YOLO | Roboflow UAV dataset |
| **DUT-Anti-UAV** | — | 5,200 | 2,600 | — | Component of mergedv3 |
| **mergedv2-cocostyle** | 2 (drone, bird) | ~4,697 | — | COCO JSON | For RF-DETR training |

---

## 3. Architectures Tested

### 3.1 Baseline Models
- **YOLO26n** — standard nano-scale YOLO26 (pretrained on COCO, yolo26n.pt)
- **YOLO26n from scratch** — same architecture, no pretrained weights
- **YOLO26s** — small-scale YOLO26
- **RF-DETR Nano** — Transformer-based detector (30.1M params)

### 3.2 Mamba (S6 SSM) Variants
All use 1D Mamba selective scan (flatten-spatial) within YOLO26 backbone:
- **yolo26-mamba** — MambaBlock at P3, P4, P5
- **yolo26-mamba2** — MambaBlock2 variant
- **yolo26-mamba-backbone1** — MambaBlock at P5 only
- **yolo26-mamba-head** — MambaBlock in FPN head only (standard backbone)

### 3.3 VMamba (Cross-Scan SSM) Variants
Use 4-way or 2-way cross-scan from VMamba library for 2D spatial awareness:
- **yolo26-v-mamba** — primary model: C3k2 + VMambaBlock at P3/P4/P5
- **yolo26-v-mamba-mk2** — C3k2 _before_ VMamba (local→global order)
- **yolo26-v-mamba-mk3** — VMamba _before_ C3k2 (global→local order)
- **yolo26-v-mamba-2way** — 2-way scan instead of 4-way
- **yolo26-v-mamba-noC3k2** — VMamba only, no C3k2 in backbone
- **yolo26-v-mamba-noC3k2-2way** — 2-way scan, no C3k2
- **yolo26-v-mamba-pure** — pure VMamba VSSBlock backbone (no CNN interleaving)
- **yolo26-v-mamba-backbone** — full VMamba pretrained backbone with Index module

### 3.4 Mamba-3 (Newly Implemented)
- **yolo26-mamba3** — VMamba3BlockYOLO using Mamba-3 SSM (trapezoidal discretization, complex-valued SSM via RoPE, MIMO)
- **yolo26-v-mamba3** — pure VMamba3 backbone variant

### 3.5 External Reference
- **Wang2024 Mamba-YOLO-T** — ported from wang2024, uses Mamba-1 selective scan with custom cross-selective-scan

---

## 4. Experimental Results

### 4.1 COCO2017 Pretraining (50 epochs)

| Model | Batch | Optimizer | LR | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|-----------|-----|-------|----------|-----------|--------|
| **YOLO26n** (baseline) | 64 | auto | 0.01 | **0.473** | **0.334** | 0.597 | 0.433 |
| **VMamba YOLO26** | 16 | SGD | 0.005 | 0.453 | 0.316 | 0.581 | 0.423 |
| **Mamba YOLO26** | 32 | SGD | 0.05 | 0.023 | 0.011 | 0.143 | 0.033 |

VMamba is competitive with baseline on COCO (−1.8pp mAP50-95). Standard Mamba (S6) with high LR completely failed.

### 4.2 mergedv2 — Drone + Bird Detection (2 classes)

#### Top Results

| Model | Pretrained? | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------------|--------|-------|----------|-----------|--------|
| **YOLO26n** (yolo26n.pt) | COCO | 40 | **0.498** | **0.255** | 0.830 | 0.458 |
| VMamba → fine-tuned from COCO | COCO | 50 | 0.443 | 0.159 | 0.790 | 0.428 |
| **YOLO26s** | No | 50 | 0.415 | 0.179 | 0.799 | 0.390 |
| Mamba (S6) backbone+head | No | 40 | 0.400 | 0.172 | 0.697 | 0.407 |
| Mamba (S6) P5-only | No | 40 | 0.400 | 0.134 | 0.775 | 0.370 |
| **VMamba (best run, data18)** | No | 50 | 0.392 | **0.188** | 0.664 | 0.403 |
| VMamba mk2 (C3k2→VMamba) | No | 50 | 0.396 | 0.170 | 0.710 | 0.351 |
| YOLO26n from scratch | No | 40 | 0.392 | 0.161 | 0.712 | 0.403 |
| VMamba 2-way | No | 50 | 0.393 | 0.142 | 0.678 | 0.364 |
| VMamba noC3k2 | No | 50 | 0.397 | 0.125 | 0.718 | 0.379 |
| VMamba mk3 (VMamba→C3k2) | No | 50 | 0.318 | 0.096 | 0.598 | 0.331 |
| VMamba pure (VSSBlock) | No | 31/50 | 0.251 | 0.086 | 0.343 | 0.390 |
| RF-DETR Nano | No | 50 | 0.250 | 0.065 | 0.642 | 0.190 |
| VMamba3 (Mamba-3 SSM) | No | 38/50 | 0.347 | 0.124 | 0.767 | 0.335 |
| Wang2024 Mamba-YOLO-T | No | 8/50 | 0.132 | 0.047 | 0.163 | 0.290 |
| Mamba head-only | No | 40 | 0.365 | 0.116 | 0.751 | 0.337 |

### 4.3 Drone-Only Datasets

| Model | Dataset | Epochs | mAP50 | mAP50-95 |
|-------|---------|--------|-------|----------|
| VMamba YOLO26 | UAV-roboflow | 13/50 | **0.759** | **0.398** |
| VMamba YOLO26 | mergedv3 | 12/50 | 0.558 | 0.253 |

VMamba shows strong performance on larger single-class drone datasets.

---

## 5. Key Findings

### 5.1 Architecture Insights
1. **Pretrained YOLO26n is the strongest baseline** on mergedv2 (mAP50-95 = 0.255), benefiting from COCO transfer learning.
2. **VMamba is competitive on COCO** (mAP50-95 = 0.316 vs 0.334 for baseline, −5.4% relative), but the gap widens on the small mergedv2 dataset where SSMs may lack enough data to learn effective scan patterns.
3. **C3k2→VMamba order (mk2) outperforms VMamba→C3k2 (mk3)** — local CNN features first, then global SSM context works better (0.170 vs 0.096 mAP50-95). This suggests SSMs are better at refining CNN features than producing features from scratch.
4. **Removing C3k2 from the backbone hurts performance**, and **pure VMamba/VSSBlock models perform worst** (~0.086 mAP50-95), confirming the CNN–SSM hybrid design is important.
5. **4-way cross-scan outperforms 2-way** (0.188 vs 0.142 mAP50-95), justifying the extra computation for richer spatial context.
6. **Standard 1D Mamba (S6) is surprisingly viable** when placed carefully — the backbone+head config (0.172) is competitive with VMamba variants, despite lacking explicit 2D spatial awareness.
7. **RF-DETR Nano (30.1M params) performed poorly** on the small mergedv2 dataset (mAP50-95 = 0.065), with highly unstable training metrics. Transformers need more data.

### 5.2 Training Sensitivity
- **Learning rate is critical for Mamba models**: SGD lr=0.05 caused complete training failure on COCO (mAP50 = 0.023), while lr=0.005 worked well.
- **SGD with lr=0.005 is the most reliable optimizer** for VMamba variants.
- **Many experiments terminated early** (drone-specific runs), suggesting training instability in some configurations.
- **VMamba3 (Mamba-3) showed instability**: one run with lr=0.01 completely failed (all zeros), while lr=0.005 yielded mAP50-95 = 0.124.

### 5.3 Model Scaling
- YOLO26s (small) on mergedv2 achieves mAP50-95 = 0.179, below pretrained YOLO26n (0.255), indicating that **pretrained weights matter more than model capacity** on small datasets.

---

## 6. Implementation Challenges

### 6.1 Mamba + CPU Compatibility
Mamba SSM kernels (selective_scan_cuda, Triton SSD) are CUDA-only. Ultralytics performs an initial CPU forward pass to compute FLOPs and parameter counts. This required adding **CPU guard logic** that returns zeros when tensors are not on CUDA, allowing the model to build successfully before moving to GPU for training.

### 6.2 AMP (Mixed Precision) dtype Mismatches
During AMP backward pass, the Triton SSD kernel received fp16 tensors for some inputs and fp32 for others (due to bias addition and normalization promoting dtypes). Fixed by explicitly casting B, C, dt to match the scan dtype before calling the kernel.

### 6.3 In-place Operations on Views
`torch.split()` returns views, and applying in-place SiLU activation on these views caused autograd errors. Fixed by cloning tensors before activation.

### 6.4 Wang2024 Mamba-YOLO API Migration
The wang2024 Mamba-YOLO implementation used deprecated `selective_scan_cuda_core` module. Ported to the current `selective_scan_cuda` API (added z parameter to fwd/bwd calls, updated AMP decorators).

### 6.5 RF-DETR Bugs
RF-DETR validation code had a NumPy indexing bug (`np.argwhere` returning 2D array instead of scalar), requiring a manual patch.

---

## 7. Mamba-3 Implementation (New)

Implemented the **Mamba-3 architecture** from "Mamba-3: Improved Sequence Modeling using State Space Principles" (ICLR 2026 Oral) with three key changes over Mamba-2:

1. **Trapezoidal Discretization** — uses a learned gate λ_t to blend current and previous states instead of zero-order hold, improving continuous-discrete approximation.
2. **Complex-Valued SSM via RoPE** — applies data-dependent Rotary Position Embeddings to B and C projections, enabling complex eigenvalues without explicit complex arithmetic.
3. **MIMO (Multi-Input Multi-Output)** — introduces configurable rank `r` allowing each SSM head to process multiple input/output channels, increasing expressiveness.

Additional features: BC bias (trainable, initialized to 1), QK-style normalization on B/C, no conv1d (removed per paper).

Model stats: **4.2M parameters, 9.0 GFLOPs** (nano scale). Initial results on mergedv2 show mAP50-95 = 0.124, with training sensitivity to learning rate.

---

## 8. Files and Structure

| File | Purpose |
|------|---------|
| `mamba_block.py` | Mamba-1 (S6) block wrappers for YOLO |
| `vmamba_block.py` | VMamba cross-scan block (SS2D + VMamba2DBlock) |
| `mamba3.py` | Mamba-3 core SSM implementation |
| `mamba3_block.py` | Mamba-3 2D vision wrappers (SS2D_Mamba3, VMamba3Block) |
| `mamba_registry.py` | Registers all custom Mamba modules with Ultralytics |
| `train.py` | Training script |
| `model-cfg/` | All YOLO model YAML configurations |
| `libs/ultralytics/` | Custom Ultralytics fork |
| `libs/VMamba/` | VMamba library with cross-scan kernels |

---

## 9. Next Steps

- [ ] Train Mamba-3 on COCO for fair comparison with VMamba/baseline
- [ ] Hyperparameter search for Mamba-3 (learning rate, d_state, expand ratio)
- [ ] Evaluate all models on mergedv3 (larger drone dataset) for more robust comparison
- [ ] Explore COCO pretraining → fine-tuning pipeline for all Mamba variants
- [ ] Test YOLO26s/m scale variants of VMamba and Mamba-3
- [ ] Compare inference speed (FPS) and memory usage across architectures


Created and tested a hybrid model, where I applied VMamba as a backbone from the new yolo26 model, reusing the yolo26 head. Tested it on Mamba1 and 2 also trying to implement it on the newest mamba3 which paper still hasn't been published, but has been aproved a month ago.
Althought vmamba1 test hasnt yealded better performance

Tests are done on the standart COCO2017 dataset and on my custom drone dataset.

For comparison other models were trained:
yolo26n, yolo26s - yolo style
RT-DETR, RF-DETR - transformers as direct competition to SSM's
YOLO-MAMBA by Wang2024 - other try of implementing ssm with older yolo8 model.


* Parašiau ir ištestavau hibridinį yolo + mamba (populiariausias SSM) modelį, kuriam pritaikiau VMamba tipo blokus kaip 'backbone', o 'head' pritaikiau iš yolo26.
* Šį hibridinį modelį ištestavau ant mamba1 ir mamba2 architektūrų, taip pat dabar bandau sukoduoti mamba3 (nes jis dar neturi implementacijos), kurio konceptas dabar yra peržiūros fazėj, ir jį aprašantis mokslinis darbas turėtų būti išleistas per ateiantį mėnesį.
* Deja mano modelis rodo 7% blogesnius rezultatus.
* Palyginimui dar treniruoju kitus modelius:
   - yolo26n, yolo26s -> yolo stiliaus
   - RT-DETR, RF-DETR -> transformeriai, kadangi naudojant SSM, mes bandom juos pakeist
   - OLO-MAMBA by Wang2024 -> ankstesnis bandymas pritaikyti Mamba prie senesnio Yolo8 modelio.
* Modeliai treniruojami naudojant standartinį COCO2017 duomenų rinkinį ir vieną jungtinį dronų, sudaryta iš kelių, siekant subalansuoti dydį, foną etc.