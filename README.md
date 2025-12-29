# DeepFake Detection using MobileNetV2

## Overview
This project aims to detect deepfake videos using **MobileNetV2** as the backbone model. The dataset used is **Celeb-DF v2**, and the pipeline includes frame extraction, preprocessing, model training, and evaluation.

---

## Project Structure
DeepFake-Detection/
│
├── data/ # Celeb-DF-v2
│ ├── Celeb-real
│ ├── Celeb-synthesis
│ 
│
├── src ├── data
        │
        ├── model_MobileNetV2/ # Model architecture, training and checkpoint scripts
        │
        ├──preprocessing/ # Scripts for frame extraction, resizing, augmentation
        │
        ├──evaluation/
        └── README.md
        └── venv

# Main libraries used:

- tensorflow / keras

- onnxruntime

- opencv-python

- numpy, pandas

- scikit-learn

- matplotlib, seaborn

- tqdm

# Dataset

- Celeb-DF v2: High-quality deepfake dataset.

- Frame extraction is done using ONNX models.

- Preprocessing steps:

- Resize frames to (160, 160) for MobileNetV2.

- Normalize pixel values.

- Balance classes using resampling


# Model Architecture

- Base model: MobileNetV2 (pretrained on ImageNet)

- Custom top layers:

- GlobalAveragePooling2D

- Dense layers with BatchNormalization and Dropout

- Final output: 1 neuron (binary classification)

- Optimizer: Adam

- Loss: Binary Crossentropy

- : L2 weight decay


# Training

- Data split: Train / Validation / Test

- Class imbalance handled with:

- Class weights or resampling

# Callbacks:

- ModelCheckpoint

- EarlyStopping

- ReduceLROnPlateau

- Data augmentation using ImageDataGenerator.


# Evaluation

* Metrics:

- Accuracy, F1-score

- ROC-AUC

- Confusion matrix visualization

- ROC curves