# Deepfake Detection using MobileNetV2

##  Introduction

This project focuses on **deepfake video detection** by combining:

* **MobileNetV2** as the main deep learning model
* **MTCNN** for face detection and frame extraction from videos
* **Celeb-DF v2 Dataset** for training and evaluation
* **Grad-CAM** for model interpretability and visualization
* **Streamlit** for deploying an interactive demo application

The goal is to build a lightweight, efficient, and explainable deepfake detection system that is easy to deploy in real-world scenarios.

---

##  Dataset

### Celeb-DF v2

* Contains both **REAL** and **FAKE (deepfake)** videos
* Widely used benchmark dataset for deepfake detection research

### Data Preprocessing

1. Load videos from the Celeb-DF v2 dataset
2. Extract frames from each video
3. Apply **MTCNN** to:

   * Detect faces
   * Crop and align facial regions
4. Resize cropped faces to **224x224**, compatible with MobileNetV2 input

### Data Balancing

* The dataset is **balanced between REAL and FAKE classes** to reduce class imbalance
---

##  Model

### MobileNetV2

* Lightweight CNN architecture optimized for speed and efficiency
* Suitable for real-time and web-based deployment

#### Overall Architecture:

* Backbone: MobileNetV2 (pretrained on ImageNet)
* Global Average Pooling
* Fully Connected Layers
* Output: Binary Classification (REAL / FAKE)

---

##  Training

* Loss Function: Binary Cross Entropy
* Optimizer: Adam
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

##  Model Evaluation

### Grad-CAM

* **Grad-CAM** is used to visualize regions of the face that the model focuses on
* Helps to:

  * Interpret model decisions
  * Identify important deepfake-related features (eyes, mouth, facial boundaries, etc.)

### Evaluation Analysis

* Compare predictions on REAL vs FAKE samples
* Analyze Grad-CAM heatmaps to verify model reliability and reasoning

---

##  Deployment with Streamlit

The Streamlit application allows users to:

* Upload a video or image
* Automatically:

  * Extract frames
  * Detect and crop faces using MTCNN
  * Predict REAL / FAKE using MobileNetV2
* Visualize:

  * Prediction results
  * Grad-CAM heatmaps

Run the application:

```bash
streamlit run app.py
```

---


##  Technologies Used

* Python
* TensorFlow / PyTorch
* OpenCV
* MTCNN
* Streamlit
* Grad-CAM

---

##  Conclusion

This project presents a complete pipeline for **Deepfake Detection**, including:

* Data preprocessing and balancing
* Training an efficient and lightweight deep learning model
* Model interpretability using Grad-CAM
* Real-world deployment via Streamlit

The system is suitable for research, demonstrations, and further extensions in practical applications.

---

