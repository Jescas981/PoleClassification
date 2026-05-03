# 🧠 Training Strategy Overview: K-Fold + Sampling + Imbalance Handling

This repository implements a robust training pipeline designed for **3D Point Cloud classification** (PLY files) using specialized techniques to handle small, imbalanced datasets.

## 📋 Table of Contents
1. [K-Fold Cross Validation](#-1-k-fold-cross-validation)
2. [Class Imbalance Management](#-2-class-imbalance-management)
3. [Sampling Strategy](#-3-weightedrandomsampler)
4. [Combined Pipeline](#-4-combined-strategy)
5. [Validation Protocol](#-5-validation-strategy)

---

## 🔁 1. K-Fold Cross Validation
Given the small size of the 3D point cloud dataset, a traditional single split is insufficient and prone to high variance.

### **The Mechanism**
The dataset is partitioned into **K equal folds**. The process iterates K times:
* **Training set:** $K-1$ folds.
* **Validation set:** 1 remaining fold.
Each sample serves as validation exactly once.

### **Why it is used here**
* **Stability:** Eliminates dependency on a single random split.
* **Reliability:** Provides a mean accuracy and standard deviation that reflect true model performance.
* **Optimization:** Maximizes the utility of every available 3D sample.

---

## ⚖️ 2. Class Imbalance Management
Point cloud datasets for infrastructure (e.g., poles) often suffer from significant class disparity:
* **Majority Class:** `monoposte`
* **Minority Class:** `biposte`

**The Risk:** Without correction, the model develops a bias toward the majority class, achieving "high accuracy" by simply ignoring the minority class.

---

## 🎯 3. WeightedRandomSampler (Batch-Level Balancing)
Instead of static **Random Oversampling** (which risks overfitting by duplicating files on disk), we utilize a **WeightedRandomSampler**.

### **Implementation Logic**
1.  **Assign Weights:** Samples are assigned weights inversely proportional to their class frequency.
    * `Minority Weight` > `Majority Weight`.
2.  **Dynamic Sampling:** During training, the Data Loader uses these weights to pull samples.
3.  **Result:** Each training batch is approximately balanced (e.g., 50% monoposte, 50% biposte) without physically duplicating data.

**Key Advantages:**
* Lower overfitting risk compared to standard oversampling.
* Dynamic variety across epochs.
* Stable gradient updates.

---

## 🔄 4. Combined Strategy
The pipeline integrates multiple layers to ensure generalization:

| Component | Purpose |
| :--- | :--- |
| **K-Fold CV** | Robust evaluation on small data. |
| **WeightedRandomSampler** | Dynamic class balancing per batch. |
| **FPS + Normalization** | Stable geometric representation of point clouds. |
| **Augmentation** | Rotation, jitter, and scaling to improve diversity. |

---

## 📊 5. Validation Strategy
A critical rule in this pipeline: **The Validation Set remains untouched (unbalanced).**

**Rationale:**
The validation set must reflect the **real-world distribution**. We do not balance it because we need to know how the model will behave in production, not just how it performs on an idealized, synthetic dataset.

---

## 📈 Expected Outputs
For every training session, the pipeline generates:
* **Per-Fold Metrics:** Individual accuracy and loss curves.
* **Final Report:** Mean Accuracy $\pm$ Standard Deviation.
* **Checkpoints:** The best-performing model for each fold.