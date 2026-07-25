# Experiment 2: Convolutional Neural Network — CIFAR-10 Classification

Image classification on CIFAR-10 using a CNN built with TensorFlow/Keras.

## Overview

- **Dataset**: CIFAR-10 (50,000 training / 10,000 test, 32×32 RGB, 10 classes)
- **Architecture**: 3 convolutional units → Flatten → 3 Dense layers
  - Unit 1: Conv(32) → Conv(32) → MaxPool → Dropout(0.5)
  - Unit 2: Conv(64) → Conv(64) → Conv(128) → MaxPool → Conv(256) → MaxPool → Dropout(0.4)
  - Dense: 512 → Dropout(0.5) → 128 → Dropout(0.5) → 10 (softmax)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Loss**: Categorical Crossentropy
- **Accuracy**: ~83% on test set

## Files

| File | Description |
|------|-------------|
| `main.py` | Simplified CNN training script (starter version) |
| `学生版.py` | Student template with TODOs for completing the network |
| `CIFAR_10_model_training.ipynb` | Full Jupyter notebook with detailed implementation |
| `check_tensorflow_install.py` | TensorFlow installation verification script |
| `cifar-10-python/` | CIFAR-10 dataset (Python format) |

## Usage

### Quick check

```bash
python check_tensorflow_install.py
```

### Run training

```bash
python main.py
```

### Jupyter notebook

```bash
jupyter notebook CIFAR_10_model_training.ipynb
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib
- Jupyter (optional, for notebook)
