# Meta-Classifier Deepfake Detection Pipeline  
A Two-Layer Ensemble System Using Multi-Encoder Feature Stacking

This repository contains the full implementation of a two-layer meta-classification architecture for deepfake detection.  
The goal of this project is to demonstrate that combining multiple encoders using a stacking-based meta-classifier can significantly outperform traditional single-model deepfake detectors.

---

## 🚀 Project Overview

### Layer 1 — Multi-Encoder Feature Extraction  
Four encoders process each face (8 frames per video), producing:  
- **4 probability scores** (one per encoder)  
- **4 × 64-dimensional feature vectors**

Encoders used:
- **ResNet50**
- **Xception**
- **ArcFace Lite** (ResNet18-based)
- **EmotionNet Lite** (ResNet18 + MLP)

Each encoder is trained with:
- Full-frame 8×224×224 crops  
- Adam optimizer  
- BCEWithLogits loss  
- K-Fold cross-validation (K=5)

The result is an **OOF (Out-of-Fold) meta-feature table**, storing predictions + 64-dim features for every video.

---

## Layer 2 — XGBoost Meta-Classifier  
The OOF table is used to train a second-layer classifier:  
- **XGBoost (binary: logistic)**  
- 800 trees  
- Max depth = 7  
- Learning rate = 0.05  
- Class imbalance handling via `scale_pos_weight`  

After training, Layer 1 is retrained on **100% of the dataset** and final predictions are passed to Layer 2.

---

## 📊 Results (Summary)

### FF++ Dataset
- **Meta-classifier AUC: ~0.986**
- Outperforms every single encoder
- t-SNE visualization shows strong class separation

### DFDC Dataset
- **Meta-classifier AUC: ~0.944**
- Also outperforms single encoders  
- Handles extreme imbalance better than direct classifiers

---

## 📁 Resources  
All code, meta-tables, model weights, and CSV outputs can be found in this repository.

---

## 📚 References  
- He et al., *Deep Residual Learning for Image Recognition* (ResNet)  
- Chollet, *Xception: Deep Learning with Depthwise Separable Convolutions*  
- Schroff et al., *FaceNet & ArcFace*  
- OpenAI, *Emotion Encoding Research*  
- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*  
- Bloom, *Stacked Generalization (Ensemble Stacking)*  
- Dosovitskiy et al., *CViT / Vision Transformers*  

---
