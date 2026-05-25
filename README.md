# EEG Emotion Recognition using Graph Convolutional Networks (GCN)

## Overview
This project implements an EEG-based Emotion Recognition System using Graph Convolutional Networks (GCNs) on the GAMEEMO dataset.  
The model represents EEG electrodes as graph nodes and learns spatial relationships between brain regions to classify emotions into four categories:

- Boring
- Calm
- Horror
- Funny

The project uses:
- EEG signal processing
- Frequency band feature extraction
- Graph-based deep learning
- PyTorch implementation of GCN

---

# Dataset

Dataset Used: GAMEEMO Dataset

Source:  
Database for Emotion Recognition System (GAMEEMO)

The dataset contains EEG recordings collected using the EMOTIV EPOC+ headset.

## EEG Channels Used

```python
CHANNELS = [
    'AF3','AF4','F3','F4','F7','F8',
    'FC5','FC6','O1','O2','P7','P8','T7','T8'
]
```

Total electrodes used: 14

---

# Emotion Labels

| Game | Emotion | Label |
|------|----------|-------|
| G1 | Boring | 0 |
| G2 | Calm | 1 |
| G3 | Horror | 2 |
| G4 | Funny | 3 |

---

# Signal Processing Pipeline

## 1. Windowing

The EEG signal is divided into overlapping windows.

```python
WINDOW_SIZE = 256
STEP_SIZE   = 128
```

At sampling frequency:

\[
f_s = 128 \text{ Hz}
\]

Each window contains:

\[
\frac{256}{128} = 2 \text{ seconds}
\]

So every graph represents:

- 2 seconds of EEG activity
- across 14 electrodes

---

# Feature Extraction

For every electrode, 9 features are extracted.

## Frequency Band Powers

Using Welch’s Power Spectral Density estimation:

\[
PSD(f)
\]

Band power is computed as:

\[
P_{band} = \int_{f_l}^{f_h} PSD(f)\,df
\]

## EEG Frequency Bands

| Band | Frequency Range | Brain Activity |
|------|-----------------|----------------|
| Delta | 0.5 – 4 Hz | Deep sleep |
| Theta | 4 – 8 Hz | Drowsiness |
| Alpha | 8 – 13 Hz | Relaxation |
| Beta | 13 – 30 Hz | Active thinking |
| Gamma | 30 – 50 Hz | High arousal |

---

## Statistical Features

Additional statistical features:

### Mean

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
\]

### Standard Deviation

\[
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
\]

### Signal Range

\[
Range = x_{max} - x_{min}
\]

### Mean Absolute Difference

\[
MAD = \frac{1}{N-1}\sum_{i=1}^{N-1}|x_{i+1}-x_i|
\]

---

# Graph Construction

Each EEG electrode is treated as a graph node.

## Graph Representation

\[
G = (V, E)
\]

Where:

- \(V\) = EEG electrodes
- \(E\) = anatomical connections between electrodes

---

# Fixed Anatomical Adjacency Matrix

A manually designed adjacency matrix is used based on neighboring electrode positions.

Example:

```python
FIXED_EDGES = [
    (0,1),(0,2),(0,4),
    (1,3),(1,5),
    ...
]
```

Self-loops are added:

\[
A = A + I
\]

---

# Adjacency Normalization

The adjacency matrix is normalized using symmetric normalization:

\[
\hat{A} = D^{-1/2} A D^{-1/2}
\]

Where:

- \(D\) = Degree matrix
- \(A\) = Adjacency matrix

This prevents feature explosion and stabilizes graph learning.

---

# Node Feature Matrix

For every graph:

\[
X \in \mathbb{R}^{14 \times 9}
\]

- 14 nodes (electrodes)
- 9 features per node

---

# Graph Convolutional Network (GCN)

## GCN Layer Formula

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

Two graph pooling methods are used.

## Mean Pooling

\[
h_{mean} = \frac{1}{N}\sum_{i=1}^{N} h_i
\]

## Max Pooling

\[
h_{max} = \max(h_i)
\]

Final pooled vector:

\[
H_{pool} = [h_{mean} \; || \; h_{max}]
\]

Dimension:

\[
128 + 128 = 256
\]

---

# Classification Head

Fully connected neural network:

\[
256 \rightarrow 128 \rightarrow 64 \rightarrow 4
\]

Final output:

- 4 emotion logits

---

# Loss Function

Cross Entropy Loss:

\[
L = -\sum_{i=1}^{C} y_i \log(\hat{y_i})
\]

Where:

- \(y_i\) = true label
- \(\hat{y_i}\) = predicted probability

---

# Optimization

Optimizer used:

```python
Adam
```

Learning rate:

\[
lr = 0.001
\]

Weight decay:

\[
10^{-4}
\]

---

# Training Details

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 100 |
| Window Size | 256 |
| Step Size | 128 |

---

# Subject-wise Split

Train/Test split is performed subject-wise.

## Training Subjects

S01 – S24

## Testing Subjects

S25 – S28

This prevents data leakage and evaluates generalization on unseen subjects.

---

# Why Subject-wise Accuracy is Lower

Random window splitting gives artificially high accuracy because windows from the same person appear in both train and test sets.

Subject-wise splitting is harder because:

- the model must generalize to completely unseen brains
- EEG signals vary strongly between individuals
- emotional EEG patterns are highly person-dependent

Therefore:

- Random split accuracy ≈ higher
- Subject-wise accuracy ≈ lower but scientifically more correct

---

# Results

## Final Test Accuracy

Approximately:

\[
30\% - 32\%
\]

on completely unseen subjects.

Random baseline:

\[
25\%
\]

for 4 classes.

---

# Important Observation

Although accuracy appears low, subject-independent EEG emotion recognition is an extremely difficult problem because:

- EEG is noisy
- emotions overlap neurologically
- different subjects produce different brain patterns
- low-cost EEG headsets have limited signal quality

Even modest improvements above random chance are meaningful.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

---

# Future Improvements

Possible improvements:

- Dynamic adjacency matrices
- Attention-based GNNs (GAT)
- Temporal modeling using LSTM/Transformer
- Better EEG preprocessing
- Subject adaptation techniques
- Frequency-domain graph learning

---

# Repository Structure

```bash
├── EEG_Emotion_GCN.ipynb
├── training_curves.png
├── electrode_graph.png
├── README.md
└── requirements.txt
```

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Notebook

Open:

```bash
EEG_Emotion_GCN.ipynb
```

in Google Colab or Jupyter Notebook.

---

# Author

Developed as a Graph Neural Network based EEG Emotion Recognition project using the GAMEEMO dataset and PyTorch.
