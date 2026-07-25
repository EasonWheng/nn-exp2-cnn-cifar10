<p align="center">
  <h1 align="center">🖼️ CNN for CIFAR-10 Image Classification<br><sub>Convolutional Neural Networks with TensorFlow/Keras</sub></h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/TensorFlow-2.14%2B-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-2.x-red?logo=keras" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Accuracy-83.1%25-success.svg" alt="Accuracy">
</p>

---

## 📖 Abstract

This project builds a **deep Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10** dataset into 10 categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The network employs multiple convolutional blocks with **batch normalization-free design**, **Dropout regularization**, and **AdamW optimization**, achieving **83.1% test accuracy** — significantly above the ~77% baseline.

## 🗂️ CIFAR-10 Dataset

| Property | Value |
|----------|-------|
| Image size | 32×32×3 (RGB) |
| Training set | 50,000 images |
| Test set | 10,000 images |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Per class | 6,000 images (5,000 train + 1,000 test) |

## 🏗️ Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       INPUT: 32×32×3                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIT 1                                                      │
│  ├─ Conv2D(32, 3×3, ReLU, same)    → 32×32×32              │
│  ├─ Conv2D(32, 3×3, ReLU, same)    → 32×32×32              │
│  ├─ MaxPool2D(2×2, same)           → 16×16×32              │
│  └─ Dropout(0.5)                                            │
├─────────────────────────────────────────────────────────────┤
│  UNIT 2                                                      │
│  ├─ Conv2D(64, 3×3, ReLU, same)    → 16×16×64              │
│  ├─ Conv2D(64, 3×3, ReLU, same)    → 16×16×64              │
│  ├─ Conv2D(128, 3×3, ReLU, same)   → 16×16×128             │
│  ├─ MaxPool2D(2×2, same)           → 8×8×128               │
│  ├─ Conv2D(256, 3×3, ReLU, same)   → 8×8×256               │
│  ├─ MaxPool2D(2×2, same)           → 4×4×256               │
│  └─ Dropout(0.4)                                            │
├─────────────────────────────────────────────────────────────┤
│  CLASSIFIER                                                  │
│  ├─ Flatten                        → 4096                   │
│  ├─ Dense(512, ReLU) + Dropout(0.5)                         │
│  ├─ Dense(128, ReLU) + Dropout(0.5)                         │
│  └─ Dense(10, Softmax)                                      │
└─────────────────────────────────────────────────────────────┘

Total params: 2,599,210  |  All trainable  |  ~9.92 MB
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| Loss | Categorical Crossentropy |
| Batch size | 128 |
| Epochs | 100 |
| Validation split | 10% |
| Data normalization | [0, 1] |

## 📁 Project Structure

```
.
├── CIFAR_10_model_training.ipynb   # 📓 Complete Jupyter notebook (recommended)
├── main.py                         # 🐍 Standalone training script
├── 学生版.py                       # 📝 Student template (with TODOs)
├── check_tensorflow_install.py     # ✅ Environment verification
└── cifar-10-python/                # 📦 CIFAR-10 dataset (Python format)
    └── cifar-10-batches-py/
        ├── data_batch_1..5         # Training batches
        ├── test_batch              # Test batch
        └── batches.meta            # Label metadata
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Verify installation
python check_tensorflow_install.py

# Expected output:
# TensorFlow version: 2.x.x
# GPU devices: [...]
# TensorFlow runtime check passed.
```

### 2. Run Training

```bash
# Quick: standalone script
python main.py

# Recommended: full notebook
jupyter notebook CIFAR_10_model_training.ipynb
```

### 3. Requirements

```bash
pip install tensorflow numpy matplotlib jupyter
```

GPU acceleration recommended (CUDA + cuDNN).

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **83.11%** |
| Baseline (teacher reference) | 76.93% |
| Improvement | +6.18% |
| Training time (NVIDIA A10) | ~15 min |
| Parameters | 2.6M |

### Key Design Choices

1. **AdamW** over Adam — decoupled weight decay improves generalization
2. **Progressive channel expansion** (32→64→128→256) — efficient feature hierarchy
3. **Moderate dropout** (0.4–0.5) — prevents overfitting without killing signal
4. **No batch normalization** — simpler architecture, competitive performance
5. **Same-padding convolutions** — preserves spatial resolution in early layers

## 📚 References

- Krizhevsky, A. "Learning Multiple Layers of Features from Tiny Images." *Technical Report*, University of Toronto, 2009.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*, 2012.
- Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." *ICLR*, 2019. (AdamW)

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

<p align="center">
  <sub>Built with TensorFlow & Keras · Part of a neural networks course series</sub>
</p>
