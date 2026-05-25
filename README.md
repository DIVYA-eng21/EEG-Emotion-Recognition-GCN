# EEG Emotion Recognition using Graph Convolutional Networks (GCN)

## Overview

This repository contains **two implementations** of EEG-based Emotion Recognition using **Graph Convolutional Networks (GCNs)** on the GAMEEMO dataset.

The project explores how brain signals from EEG electrodes can be represented as graphs and processed using deep learning to classify emotional states.

The two notebooks demonstrate:

1. **Dynamic Correlation-based GCN** EEG_Emotion_GCN
2. **Fixed Anatomical Adjacency GCN** EEG_Emotion_Analysiss_SW

Both approaches use:
- EEG signal preprocessing
- Frequency-domain feature extraction
- Graph convolutional neural networks
- Emotion classification using PyTorch

---

# Repository Structure

```bash
├── EEG_Emotion_GCN.ipynb
├── EEG_Emotion_Analysiss_SW.ipynb
├── README.md
```

---

# Notebook 1 — EEG_Emotion_GCN

## Idea

In this approach:

- Every EEG window is converted into a graph.
- EEG electrodes are treated as graph nodes.
- Connections between nodes are generated dynamically using signal correlation.

The adjacency matrix changes for every EEG sample depending on brain activity.

---

## Dynamic Graph Construction

Correlation between electrode signals:
A_{ij} = corr(x_i, x_j)

If correlation exceeds threshold:
corr=0.5
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

40\% - 41\%

However, this higher accuracy is partially due to random window splitting, where windows from the same subjects appear in both train and test sets.
This can cause information leakage.

---

# Notebook 2 —EEG_Emotion_Analysiss_SW

## Idea

In this implementation:

- EEG electrodes are connected using fixed anatomical relationships.
- The adjacency matrix remains constant for all samples.

The graph structure is manually designed according to electrode placement on the scalp.

---

# Fixed Anatomical Graph

Graph:

G = (V,E)

Where:
- V = EEG electrodes
- E = anatomical neighboring connections

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

hat{A} = D^{-1/2} A D^{-1/2}

Where:
- A = adjacency matrix
- D = degree matrix
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


# EEG Signal Processing

## EEG Channels Used

```python
CHANNELS = [
    'AF3','AF4','F3','F4','F7','F8',
    'FC5','FC6','O1','O2','P7','P8','T7','T8'
]
```

Total electrodes:14


# Windowing

EEG signals are divided into overlapping windows.

```python
WINDOW_SIZE = 256
STEP_SIZE   = 128
```

Sampling Frequency:
f_s = 128Hz

Each graph represents:
- 2 seconds of EEG activity
- across 14 electrodes

---

# Feature Extraction

For every electrode, 9 features are extracted.

---




- 14 nodes
- 9 features per node
# Graph Convolutional Network (GCN)

The model uses Graph Convolutional Networks to learn spatial relationships between EEG electrodes.

Each EEG electrode is treated as a graph node, and the adjacency matrix defines how electrodes exchange information with neighboring electrodes.

---

## GCN Layer Equation

H^{(l+1)} ={ReLU}(hat{A}H^{(l)}W^{(l)}

Where:

- (H^{(l)}) → node feature matrix at layer \(l\)
- (hat{A}) → normalized adjacency matrix
- (W^{(l)}) → learnable weight matrix
- ReLU → activation function

---

## What Happens Inside a GCN Layer

### Step 1 — Neighbor Aggregation

The adjacency matrix aggregates information from neighboring electrodes:

Single GCN layer:
        H_out = ReLU( A_norm @ H_in @ W )
    A_norm aggregates neighbor features,
    W transforms them — both learned end-to-end.
This allows each electrode to receive information from connected brain regions.
---

### Step 2 — Feature Transformation

The aggregated features are transformed using learnable weights.
The network learns meaningful brain representations during training.

---

### Step 3 — Non-linearity

ReLU activation introduces non-linearity.
This helps the network learn complex emotional patterns.

---

# Classification Head

The pooled graph representation is passed through fully connected layers.

Final output:
- 4 emotion logits corresponding to:
  - Boring
  - Calm
  - Horror
  - Funny

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn

---

# Scientific Observation

EEG emotion recognition is a highly difficult problem because:

- EEG signals are noisy
- emotions overlap neurologically
- brain patterns differ across subjects

Therefore:
- subject independent testing is significantly harder
- lower accuracy is expected

---

# Dataset Download

The dataset is downloaded automatically using the Kaggle API.

You must upload:

```bash
kaggle.json
```

Download it from:

Kaggle → Account → Create New API Token

---


---

