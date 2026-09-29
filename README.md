# SE4050 – Brain Tumor Classification

This repository contains the implementation, experiments, trained-model outputs, and evaluation results for the **SE4050 – Deep Learning** group assignment.

The project investigates the classification of brain MRI images into four categories using and comparing four distinct deep learning architectures:

- Custom CNN
- MobileNetV2
- ResNet50
- EfficientNetB0

The objective is to evaluate the models under a common classification task and compare their predictive performance, model complexity, training behaviour, and computational characteristics.

---

## 1. Problem Description

Brain MRI classification is an image-classification problem in which MRI scans are assigned to different diagnostic categories.

In this project, the models classify MRI images into the following four classes:

| Class ID | Class |
|----------|-------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

Four different deep learning architectures were implemented and evaluated to investigate how different network designs perform on the same brain MRI classification problem.

---

## 2. Dataset

The project uses the **Brain Tumor MRI Dataset** provided through Kaggle.

### Dataset Source

Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The original dataset should be downloaded from Kaggle before running the notebooks.

The dataset is **not included in this repository** because of its size. The dataset-access link above provides access to the original images.

### Dataset Classes

The dataset contains four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project uses predefined training, validation, and test splits for model development and evaluation.

The test set is kept separate from model development and is used only for the final evaluation.

---

## 3. Repository Structure

The repository is organized as follows:

```text
SE4050-Brain-Tumor-Classification/
│
├── dataset/
│   └── splits/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── notebooks/
│   ├── 01_CustomCNN.ipynb
│   ├── 02_MobileNetV2.ipynb
│   ├── 03_ResNet50.ipynb
│   └── 04_EfficientNetB0.ipynb
│
├── results/
│   ├── CustomCNN/
│   ├── MobileNetV2/
│   ├── ResNet50/
│   └── EfficientNetB0/
│
├── .gitignore
├── README.md
└── requirements.txt
