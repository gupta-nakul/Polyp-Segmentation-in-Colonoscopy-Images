# Polyp Segmentation in Colonoscopy Images using Deep Learning

## 📌 Overview

This repository presents an advanced **polyp segmentation system** for **colonoscopy images**, leveraging **deep learning models** to enhance the early detection and diagnosis of **colorectal cancer**. The project implements **DeepLabV3+ (ResNet50)** and **ResUNet++**, incorporating **stochastic activation functions** and **custom data augmentation** techniques to **improve segmentation accuracy and generalization**.

The **primary goal** is to accurately **segment polyps from colonoscopy images**, mitigating challenges such as variability in polyp appearance, size, shape, and resemblance to benign structures.

---

## 🔍 Problem Statement

Colorectal cancer is a leading cause of cancer-related deaths globally. **Accurate and automated polyp segmentation** in colonoscopy images is crucial for early detection and prevention. The primary challenges in this task include:

- **High variability in polyp appearance** (size, shape, and color)
- **Low contrast between polyps and surrounding tissue**
- **Presence of artifacts and noise in endoscopic imaging**

This project aims to **enhance segmentation accuracy** by employing **deep learning models** optimized with **custom augmentations** and **stochastic activation functions**.

---

## 📊 Dataset

We utilize the **Kvasir-SEG dataset**, a publicly available dataset for medical image segmentation. It consists of:

- **1,000 high-quality polyp images**
- **Expert-annotated ground truth segmentation masks**
- **Image resolutions ranging from 332×487 to 1920×1072 pixels**

### 🔄 **Data Augmentation**
To increase dataset diversity and improve generalization, we apply the following augmentation techniques:
- **Random Crop, Center Crop**
- **Horizontal & Vertical Flip**
- **Cutout & Scale Augmentation**
- **Random Rotation**
- **Brightness Augmentation**
- **RGB to Grayscale Conversion**

Each image undergoes **31 unique transformations**, expanding the dataset to **24,800 training samples**.

---

## 🏗️ Methodology

### **1️⃣ Model Architectures**
#### **DeepLabV3+ (ResNet50 Backbone)**
- **Atrous Convolution**: Captures multi-scale information without reducing resolution.
- **ASPP (Atrous Spatial Pyramid Pooling)**: Enhances feature extraction across different scales.
- **Encoder-Decoder Structure**: Refines spatial boundaries for **accurate segmentation**.

#### **ResUNet++**
- **Residual Blocks**: Prevents vanishing gradients, allowing deeper networks.
- **Attention Gates**: Enhances focus on polyp regions for improved accuracy.
- **Multi-scale Feature Aggregation**: Extracts features across multiple resolutions.

### **2️⃣ Custom Enhancements**
#### **📌 Stochastic Activation Functions**
Inspired by recent studies, we introduce **random activation selection** for specific layers, including:
- **ReLU, Leaky ReLU, ELU, PReLU, SReLU, MeLU, GaLU, Mish**
- Random activations are **dynamically assigned per training cycle**, improving **generalization**.

#### **📌 Ensemble Learning**
- **Five variations** of each model are trained with different activation functions.
- The models' **softmax outputs are averaged** to improve performance and robustness.

---

## 📏 Evaluation Metrics

To assess segmentation accuracy, we utilize:

| Metric | Description |
|--------|------------|
| **Dice Coefficient (DSC)** | Measures segmentation overlap between prediction and ground truth. |
| **Intersection over Union (IoU)** | Also known as Jaccard Index, quantifies the accuracy of segmentation. |
| **Precision & Recall** | Evaluate false positives and false negatives. |

---

## 🏆 Results

| Model | Dice Coefficient (DSC) ↑ | IoU (Jaccard Index) ↑ | Precision ↑ | Recall ↑ |
|--------|----------------|----------------|------------|------------|
| **DeepLabV3+** | 0.7231 | 0.7063 | 0.6571 | 0.7731 |
| **DeepLabV3+ (RandAct)** | 0.7852 | 0.7669 | 0.7135 | 0.8395 |
| **ResUNet++** | 0.8132 | 0.7943 | 0.7389 | 0.8695 |
| **ResUNet++ (RandAct)** | **0.8347** | **0.8153** | **0.7566** | **0.8924** |

**Key Findings:**
- **Randomized activation functions** significantly improved segmentation performance.
- **ResUNet++ with stochastic activations** achieved the **highest DSC and IoU** scores.
- **Custom augmentation techniques** played a crucial role in improving model generalization.

