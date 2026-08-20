# Automated Corrosion Detection in the Energy Sector

Coursework for the "Computer Vision for the Energy Sector" course. This project implements and benchmarks classical machine learning and deep learning workflows to detect corrosion on industrial assets, focusing on resolving domain shift challenges in underwater environments.

## Project Structure & Results

### Task 1: Surface Corrosion Detection (Benchmarking)
Comparison of classical ML against a custom CNN on surface imagery.
*   **Dataset:** 1,232 surface images (Grayscale, 128×128). Imbalanced dataset (mostly positive samples).
*   **Model 1: SVM + HOG**
    *   Features: Histogram of Oriented Gradients (HOG).
    *   Classifier: Linear Support Vector Classifier (SVC).
    *   **Metrics:** Accuracy: 87% | **ROC AUC: 0.76**
*   **Model 2: Custom CNN**
    *   Architecture: 2x Conv2D + MaxPooling layers, followed by Dense layers.
    *   **Metrics:** Accuracy: 89% | **ROC AUC: 0.62** (biased due to class imbalance)

### Task 2: Cross-Domain Evaluation (Domain Shift)
Evaluating the surface-trained Linear SVM model directly on an underwater environment without adaptation.
*   **Dataset:** 51 underwater images (24 positive, 27 negative).
*   **Metrics:** Accuracy: 49% | **ROC AUC: 0.44**
*   **Result:** Catastrophic performance drop due to domain shift (water turbidity, lighting, and texture changes).

### Task 3: Underwater Corrosion Detection (Domain Adaptation)
Resolving domain shift using contrast enhancement, dataset expansion, and transfer learning.
*   **Dataset Expansion:** Dataset increased to 206 underwater images (129 negative, 77 positive).
*   **Preprocessing:** Applied Contrast Limited Adaptive Histogram Equalization (CLAHE) in the LAB color space to recover fine surface textures under water.
*   **Model:** Pretrained VGG16 (frozen base) + GlobalAveragePooling2D + Dense head.
*   **Validation:** 5-Fold Stratified Cross-Validation.
*   **Metrics:** Accuracy: 77% | **ROC AUC: 0.86**

---

## Tech Stack
- Python 3
- OpenCV & Scikit-Image (Preprocessing & HOG)
- TensorFlow & Keras (CNN / VGG16)
- Scikit-Learn (SVM & Evaluation Metrics)
- Matplotlib

## Quick Start

1. Install dependencies:
```bash
pip install tensorflow scikit-learn scikit-image opencv-python matplotlib numpy
```

2. Run the pipeline:
```bash
jupyter notebook corrosion_vision_energy.ipynb
```
