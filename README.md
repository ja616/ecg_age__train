# SSSD-ECG
### Age Conditioned Diffusion Model for Synthetic 12 Lead ECG Generation

> Conditional Diffusion with Structured State Space (S4) Networks for Age Aware ECG Synthesis using the PTB-XL Dataset.

---

## Overview

SSSD-ECG is a conditional diffusion framework for generating realistic synthetic electrocardiograms (ECGs) conditioned on patient age.

The model learns the temporal characteristics of ECG waveforms using Structured State Space (S4) layers while leveraging Denoising Diffusion Probabilistic Models (DDPMs) to progressively synthesize high fidelity cardiac signals.

Rather than directly generating all twelve ECG leads, the framework models eight physical leads and reconstructs the remaining four leads using standard clinical ECG relationships, producing complete 12 lead ECG recordings suitable for visualization and downstream research.

---

# Problem Statement

Large scale ECG datasets often suffer from demographic imbalance, particularly across age groups.

This imbalance affects the robustness and fairness of machine learning models trained on clinical ECG data.

The objective of this project is to learn the age dependent distribution of ECG signals and generate realistic synthetic ECGs that can improve representation of underrepresented age groups without compromising physiological characteristics.

---

# Objectives

• Generate realistic synthetic ECG signals conditioned on patient age

• Improve representation of underrepresented age groups

• Learn long range temporal dependencies using S4 layers

• Preserve clinically meaningful ECG morphology

• Produce complete 12 lead ECGs through lead reconstruction

---

# Dataset

Dataset: **PTB-XL**

Input ECG Shape

```
(N,1000,12)
```

Age Labels

```
(N,)
```

Sampling Frequency

```
100 Hz
```

Condition Variable

```
Patient Age
```

---

# Methodology

The overall workflow consists of five stages.

```
PTB-XL Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Conditional DDPM Training
        │
        ▼
Reverse Diffusion Sampling
        │
        ▼
12 Lead ECG Reconstruction
```

---

# Data Pipeline

The preprocessing pipeline performs:

- ECG loading
- Age normalization
- Eight lead selection
- Train validation test split
- DataLoader creation

Selected Model Leads

```
Lead I
V1
V2
V3
V4
V5
V6
aVF
```

Remaining Leads reconstructed

```
Lead II
Lead III
aVR
aVL
```

using linear ECG relationships.

---

# Model Architecture
<img width="1536" height="1024" alt="ChatGPT Image Jul 14, 2026, 08_15_58 AM" src="https://github.com/user-attachments/assets/b95ffb01-e516-442f-8eb0-a1d823a9e3e5" />

The proposed SSSD-ECG architecture combines

- Conditional DDPM
- Structured State Space (S4) Layers
- Residual Learning
- Skip Connections
- Age Conditioning

### Input

```
Noisy ECG (8 × 1000)

+

Age Embedding

+

Diffusion Timestep Embedding
```

↓

Initial Convolution

↓

36 Residual Blocks

↓

Each Residual Block

```
1×1 Convolution

↓

S4 Layer

↓

S4 Layer

↓

Gated Activation

↓

Residual Connection

↓

Skip Connection
```

↓

Skip Aggregation

↓

Final Convolution

↓

Predicted Noise

---

# Training Pipeline

During training,

1. Sample a clean ECG
2. Randomly select a diffusion timestep
3. Add Gaussian noise
4. Predict the injected noise
5. Compute denoising loss
6. Update model parameters

```
Real ECG

↓

Forward Diffusion

↓

Noisy ECG

↓

SSSD ECG

↓

Noise Prediction

↓

MSE Loss

↓

Backpropagation

↓

Checkpoint
```

---

# Inference Pipeline

Given an arbitrary age,

the model generates synthetic ECGs through iterative reverse diffusion.

```
Desired Age

↓

Age Embedding

↓

Random Gaussian Noise

↓

Reverse Diffusion

↓

Synthetic 8 Lead ECG

↓

Lead Reconstruction

↓

Synthetic 12 Lead ECG
```

---

# Hyperparameters

| Parameter | Value |
|------------|------:|
| Diffusion Steps | 200 |
| Residual Blocks | 36 |
| Residual Channels | 256 |
| Skip Channels | 256 |
| S4 State Dimension | 64 |
| S4 Maximum Length | 1000 |
| Age Embedding | 128 |
| Learning Rate | 2e-4 |
| Batch Size | 8 |
| Training Iterations | 100000 |
| Optimizer | Adam |

---

# Evaluation & Testing

The generated ECGs are evaluated using qualitative and conditional generation analysis.

## Visual ECG Comparison

The repository provides side by side visualization between

- Real ECG
- Generated ECG

using age matched samples from PTB-XL.

Visualization follows a standard clinical twelve lead ECG layout with ECG paper style grids, enabling inspection of

- Morphological similarity
- Lead consistency
- Waveform continuity
- Age appropriate characteristics

---

## Standard Inference

The inference pipeline

- Loads held out test ages
- Generates age conditioned ECGs
- Saves generated signals together with normalized and original age labels

---

## Elderly Population Analysis

A dedicated evaluation pipeline generates synthetic ECGs for ages

```
60 – 110 years
```

allowing targeted analysis of elderly cardiac signal generation.

---

## Data Validation

The preprocessing stage validates

- ECG tensor dimensions
- Label alignment
- Age normalization
- Dataset consistency
- Lead selection

before model training.

---

# Results

The trained model successfully learns the age conditioned distribution of ECG signals and generates physiologically plausible synthetic ECGs across diverse age groups.

The generation pipeline preserves:

- Temporal waveform continuity
- Multi lead consistency
- Age dependent morphological variations
- Clinically interpretable ECG structure

Visual comparison with PTB-XL samples demonstrates realistic signal morphology while maintaining coherent lead relationships after twelve lead reconstruction.

---

# Repository Structure

```
├── train.py
├── inference.py
├── condition_inference_60+.py
├── generate.py
├── ecg_data_pre.py
├── configs/
├── checkpoints/
├── utils/
└── SSSD_ECG/
```

---

# Future Work

- Quantitative evaluation using distribution based metrics
- Downstream ECG classification validation
- Multi condition generation
- Disease conditioned ECG synthesis
- Clinical validation with cardiologists

---

# Acknowledgements

Dataset

- PTB-XL

Frameworks

- PyTorch
- NumPy
- Matplotlib

Research Area

- Generative AI
- Diffusion Models
- Biomedical Signal Processing
- Time Series Modeling
