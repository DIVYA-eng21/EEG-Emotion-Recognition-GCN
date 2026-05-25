# EEG Emotion Recognition using Graph Convolutional Networks (GCN)

## Overview

This repository contains **two implementations** of EEG-based Emotion Recognition using **Graph Convolutional Networks (GCNs)** on the GAMEEMO dataset.

The project explores how brain signals from EEG electrodes can be represented as graphs and processed using deep learning to classify emotional states.

The two notebooks demonstrate:

1. **Dynamic Correlation-based GCN**
2. **Fixed Anatomical Adjacency GCN**

Both approaches use:
- EEG signal preprocessing
- Frequency-domain feature extraction
- Graph neural networks
- Emotion classification using PyTorch

---

# Repository Structure

```bash
├── EEG_Emotion_GCN_DynamicAdj.ipynb
├── EEG_Emotion_GCN_FixedAdj.ipynb
├── training_curves.png
├── electrode_graph.png
├── README.md
└── requirements.txt
```

---

# Notebook 1 — Dynamic Correlation-based GCN

## File
```bash
EEG_Emotion_GCN_DynamicAdj.ipynb
```

## Idea

In this approach:

- Every EEG window is converted into a graph.
- EEG electrodes are treated as graph nodes.
- Connections between nodes are generated dynamically using signal correlation.

The adjacency matrix changes for every EEG sample depending on brain activity.

---

## Dynamic Graph Construction

Correlation between electrode signals:

\[
A_{ij} = corr(x_i, x_j)
\]

If correlation exceeds threshold:

\[
|A_{ij}| > \tau
\]

then an edge is created.

This creates:
- sample-specific graphs
- adaptive brain connectivity

---

## Advantages

- Captures changing brain relationships
- More adaptive
- Learns subject-specific connectivity

---

## Disadvantages

- More noisy
- Correlation may not represent true brain connectivity
- Possible overfitting
- Computationally expensive

---

## Results

Approximate Accuracy:

\[
40\% - 41\%
\]

However, this higher accuracy is partially due to random window splitting, where windows from the same subjects appear in both train and test sets.

This can cause information leakage.

---

# Notebook 2 — Fixed Anatomical Adjacency GCN

## File
```bash
EEG_Emotion_GCN_FixedAdj.ipynb
```

## Idea

In this implementation:

- EEG electrodes are connected using fixed anatomical relationships.
- The adjacency matrix remains constant for all samples.

The graph structure is manually designed according to electrode placement on the scalp.

---

# Fixed Anatomical Graph

Graph:

\[
G = (V,E)
\]

Where:
- \(V\) = EEG electrodes
- \(E\) = anatomical neighboring connections

Example:

```python
FIXED_EDGES = [
    (0,1),(0,2),(0,4),
    (1,3),(1,5),
    ...
]
```

---

# Adjacency Normalization

Normalized adjacency matrix:

\[
\hat{A} = D^{-1/2} A D^{-1/2}
\]

Where:
- \(A\) = adjacency matrix
- \(D\) = degree matrix

This stabilizes graph convolution.

---

# Why Fixed Graph?

This approach avoids:
- noisy correlation edges
- unstable graph structures
- information leakage from dynamic graphs

It focuses on:
- stable anatomical brain structure
- better scientific validity

---

# Subject-wise Train/Test Split

Unlike the first notebook, this implementation uses:

## Subject-independent evaluation

Training Subjects:
- S01 → S24

Testing Subjects:
- S25 → S28

This ensures:
- completely unseen subjects during testing
- more realistic EEG evaluation

---

# Why Accuracy Became Lower

The first implementation used random window splitting.

That means:
- windows from the same person appeared in both train and test sets
- the model partially memorized subject-specific patterns

The second implementation avoids this leakage.

Therefore:
- harder task
- more realistic evaluation
- lower but scientifically correct accuracy

---

# Final Accuracy Comparison

| Model | Graph Type | Split Type | Accuracy |
|------|------|------|------|
| Dynamic GCN | Correlation-based | Random Window Split | ~41% |
| Fixed GCN | Anatomical Graph | Subject-wise Split | ~30% |

---

# EEG Signal Processing

## EEG Channels Used

```python
CHANNELS = [
    'AF3','AF4','F3','F4','F7','F8',
    'FC5','FC6','O1','O2','P7','P8','T7','T8'
]
```

Total electrodes:
\[
14
\]

---

# Windowing

EEG signals are divided into overlapping windows.

```python
WINDOW_SIZE = 256
STEP_SIZE   = 128
```

Sampling Frequency:

\[
f_s = 128Hz
\]

Window duration:

\[
\frac{256}{128} = 2 \text{ seconds}
\]

Each graph represents:
- 2 seconds of EEG activity
- across 14 electrodes

---

# Feature Extraction

For every electrode, 9 features are extracted.

---

# Frequency Band Powers

Using Welch’s Power Spectral Density:

\[
PSD(f)
\]

Band power:

\[
P_{band} = \int_{f_l}^{f_h} PSD(f)\,df
\]

---

# EEG Frequency Bands

| Band | Frequency |
|------|------|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–50 Hz |

---

# Statistical Features

Additional features:

## Mean
\[
\mu = \frac{1}{N}\sum x_i
\]

## Standard Deviation
\[
\sigma = \sqrt{\frac{1}{N}\sum(x_i-\mu)^2}
\]

## Range
\[
Range = x_{max} - x_{min}
\]

## Mean Absolute Difference
\[
MAD = \frac{1}{N-1}\sum |x_{i+1}-x_i|
\]

---

# Node Feature Matrix

For every graph:

\[
X \in \mathbb{R}^{14 \times 9}
\]

Where:
- 14 nodes
- 9 features per node

---

# Graph Convolutional Network

## GCN Equation

\[
H^{(l+1)} = ReLU(\hat{A}H^{(l)}W^{(l)})
\]

Where:
- \(H^{(l)}\) = node features
- \(\hat{A}\) = normalized adjacency matrix
- \(W^{(l)}\) = learnable weights

---

# Model Architecture

## GCN Layers

### Layer 1
\[
9 \rightarrow 64
\]

### Layer 2
\[
64 \rightarrow 64
\]

### Layer 3
\[
64 \rightarrow 128
\]

---

# Pooling

## Mean Pooling

\[
h_{mean} = \frac{1}{N}\sum h_i
\]

## Max Pooling

\[
h_{max} = \max(h_i)
\]

Combined representation:

\[
H_{pool} = [h_{mean} || h_{max}]
\]

Final pooled dimension:

\[
256
\]

---

# Classification Head

Fully connected layers:

\[
256 \rightarrow 128 \rightarrow 64 \rightarrow 4
\]

Output:
- 4 emotion logits

---

# Loss Function

Cross Entropy Loss:

\[
L = -\sum y_i \log(\hat{y_i})
\]

---

# Optimizer

Adam Optimizer:

```python
Adam(model.parameters(), lr=0.001)
```

Weight decay:

\[
10^{-4}
\]

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

---

# Important Scientific Observation

EEG emotion recognition is extremely difficult because:
- EEG signals are noisy
- emotions overlap neurologically
- different subjects have different brain patterns
- low-cost EEG devices have limited precision

Therefore:
- subject-wise evaluation is much harder
- lower accuracy is expected
- realistic evaluation is more important than inflated accuracy

---

# Future Improvements

Possible future work:
- Graph Attention Networks (GAT)
- Temporal GNNs
- Transformer-based EEG modeling
- Dynamic temporal adjacency learning
- Subject adaptation techniques
- Functional connectivity analysis

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Notebook

Open either notebook:

```bash
EEG_Emotion_GCN_DynamicAdj.ipynb
```

or

```bash
EEG_Emotion_GCN_FixedAdj.ipynb
```

using:
- Google Colab
- Jupyter Notebook

---

# Dataset Download

Dataset automatically downloads using Kaggle API.

You must upload:

```bash
kaggle.json
```

from your Kaggle account.

Path:
Kaggle → Account → Create New API Token

---

# Author

Graph Neural Network based EEG Emotion Recognition project implemented using PyTorch and the GAMEEMO dataset.
