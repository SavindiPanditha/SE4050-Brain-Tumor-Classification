# SE4050 – Brain Tumor Classification Using MRI Images

**Module:** SE4050 – Deep Learning (2026) | **Group:** SE4050_G12
**Institution:** Sri Lanka Institute of Information Technology (SLIIT), Department of Software Engineering

This repository contains the source code, experiments, saved outputs and evaluation results for the SE4050 Deep Learning group assignment. The project classifies brain MRI images into four classes using **four distinct deep-learning architectures** under a common experimental framework:

- Custom CNN (trained from scratch)
- MobileNetV2 (transfer learning)
- ResNet50 (transfer learning + fine-tuning experiment)
- EfficientNetB0 (transfer learning)

The aim is not to build a single model, but to compare predictive performance, class-level errors, generalization, training stability, model complexity and computational cost on the same problem.

> **Disclaimer:** This is an academic experiment. The results do not validate any model for clinical diagnosis.

---

## 1. Group Members and Contributions

| Student ID | Name | Model / Area |
|---|---|---|
| IT23257672 | D.M.B. Charuna | Custom CNN + common dataset preparation |
| IT23201514 | Perera M.E.N. | MobileNetV2 |
| IT23149908 | Udawatta V.D. | ResNet50 (frozen backbone + conv5 fine-tuning) |
| IT23357976 | Panditha S.D. | EfficientNetB0 |

Contribution history is traceable through the commit history, branches and pull requests of this repository.

---

## 2. Problem Description

A supervised multi-class image-classification task: given a brain MRI image, predict one of four classes.

| Label | Class | Folder name |
|---|---|---|
| 0 | Glioma | `glioma` |
| 1 | Meningioma | `meningioma` |
| 2 | No Tumor | `notumor` |
| 3 | Pituitary | `pituitary` |

Labels are defined once in shared CSV files and read by all four notebooks, so class indices are identical across experiments.

---

## 3. Dataset

- **Name:** Brain Tumor MRI Dataset
- **Creator:** Masoud Nickparvar
- **Source (Kaggle):** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Licence:** see the licence shown on the Kaggle dataset page.
- **Type:** MRI images (JPG) in four classes, supplied in `Training/` and `Testing/` directories.

The dataset is **not included** in this repository. Download it from Kaggle and place it in a folder of your choice (see Section 6).

### Class-balanced dataset used in this project

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Glioma | 1,120 | 280 | 400 | 1,800 |
| Meningioma | 1,120 | 280 | 400 | 1,800 |
| No Tumor | 1,120 | 280 | 400 | 1,800 |
| Pituitary | 1,120 | 280 | 400 | 1,800 |
| **Total** | **4,480** | **1,120** | **1,600** | **7,200** |

### Splitting strategy

- The original `Testing` directory is kept unchanged as a fixed **test set** (400 images per class) and is used **only for final evaluation**.
- The original `Training` directory (5,600 images) is split **80:20 (stratified)** into training and validation sets using **random seed 42**.
- The splits are stored in three shared CSV files: `train.csv`, `val.csv`, `test.csv`.
  Each row has the fields `filepath, label, class_name`, for example:
  `Training/pituitary/Tr-pi_135.jpg, 3, Pituitary`
- Checks performed: all paths exist, labels map consistently, class counts are as expected, and **there is no overlap** between train, validation and test file lists (no data leakage).

---

## 4. Repository Structure

```text
SE4050-Brain-Tumor-Classification/
│
├── dataset/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
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
```

---

## 5. Development Environment

- Python 3.x
- Google Colab (GPU runtime) / Jupyter Notebook
- TensorFlow / Keras, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pillow

All dependencies are listed in `requirements.txt`.

---

## 6. Installation and Setup

**Step 1 – Clone the repository**

```bash
git clone https://github.com/SavindiPanditha/SE4050-Brain-Tumor-Classification.git
cd SE4050-Brain-Tumor-Classification
```

**Step 2 – Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 – Download the dataset**

Download the dataset from the Kaggle link in Section 3 and extract it so that it contains the `Training/` and `Testing/` folders.

**Step 4 – Set the dataset path**

Each notebook has a configuration cell near the top. Set the dataset root path there (for example, your Google Drive path in Colab). The CSV files in `dataset/splits/` store **relative** paths, which are joined to this dataset root when the `tf.data` pipeline is built.

---

## 7. Running the Experiments

Open and run the notebooks **in order, cell by cell**. Each notebook performs preprocessing, training, validation, final test evaluation and saves its outputs.

| Notebook | Model |
|---|---|
| `notebooks/01_CustomCNN.ipynb` | Custom CNN |
| `notebooks/02_MobileNetV2.ipynb` | MobileNetV2 |
| `notebooks/03_ResNet50.ipynb` | ResNet50 |
| `notebooks/04_EfficientNetB0.ipynb` | EfficientNetB0 |

---

## 8. Experimental Configuration

### Common settings (all models)

| Setting | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 16 |
| Maximum epochs | 10 |
| Random seed | 42 |
| Loss | Sparse categorical cross-entropy |
| Optimizer | Adam |
| Output layer | 4-unit Softmax |
| Training-set shuffling | Yes (buffer = 4,480, seed 42); validation/test not shuffled |
| Final evaluation | Same untouched 1,600-image test set for all models |

### Model-specific settings

| Model | Learning rate | Preprocessing | Augmentation | Training strategy |
|---|---|---|---|---|
| Custom CNN | 0.001 | Pixels ÷ 255 (0–1) | None | Trained from scratch, 10 epochs, no early stopping |
| MobileNetV2 | 0.0001 | MobileNetV2 `preprocess_input` (−1 to 1) | None | Frozen ImageNet backbone; checkpointing + early stopping on validation accuracy (patience 3) |
| ResNet50 | 0.001 (Stage 1); 0.00001 (fine-tuning) | ResNet50 `preprocess_input` (BGR, mean-centred) | Flip, rotation 0.05, zoom 0.1 | Frozen backbone, then conv5 fine-tuning (BatchNorm frozen); early stopping + checkpointing on validation loss |
| EfficientNetB0 | 0.001 | Built-in rescaling inside the model (raw 0–255 input) | Flip, rotation 0.05, zoom 0.1 | Frozen ImageNet backbone; early stopping on validation loss, checkpointing on validation accuracy |

Augmentation is applied to **training images only**.

> **Note:** Preprocessing, learning rate and augmentation differ between models because the pretrained networks require architecture-specific inputs. Performance differences therefore cannot be attributed to architecture alone (see the report, Section 8).

---

## 9. Model Architectures (Summary)

| Model | Type | Transfer learning | Classifier head | Total params | Trainable params |
|---|---|---|---|---|---|
| Custom CNN | 3 × (Conv + MaxPool) → Flatten | No | Dense(128, ReLU) → Dropout(0.5) → Dense(4) | 12,938,948 | 12,938,948 |
| MobileNetV2 | Lightweight inverted-residual CNN | Yes | GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(4) | 2,422,468 | 164,484 |
| ResNet50 | Deep residual CNN | Yes | GAP → Dropout(0.3) → Dense(4) | 23,595,908 | 8,196 (frozen stage) |
| EfficientNetB0 | Compound-scaled CNN | Yes | GAP → Dropout(0.3) → Dense(4) | 4,054,695 | 5,124 |

---

## 10. Results (Final Test Set – 1,600 images)

| Model | Test Accuracy | Test Loss | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| Custom CNN | 88.00% | 1.3299 | 88.21% | 88.00% | 87.77% |
| MobileNetV2 | 87.56% | 0.4534 | 88.66% | 87.56% | 87.33% |
| ResNet50 (frozen backbone) | 88.06% | 0.4249 | 89.18% | 88.06% | 87.99% |
| EfficientNetB0 | 82.25% | 0.5072 | 82.94% | 82.25% | 81.47% |

### Computational comparison

| Model | Training time | Total params | Trainable params |
|---|---|---|---|
| Custom CNN | 26.95 min | 12,938,948 | 12,938,948 |
| MobileNetV2 | 11.6 min | 2,422,468 | 164,484 |
| ResNet50 | 6.3 min | 23,595,908 | 8,196 |
| EfficientNetB0 | 4.86 min | 4,054,695 | 5,124 |

Training times were measured in individual Colab notebooks and are not a controlled hardware benchmark.

### Key observations

- All models found **Glioma vs Meningioma** the hardest pair to separate. **No Tumor** and **Pituitary** were the easiest classes.
- The Custom CNN showed the clearest overfitting at the end of training (training accuracy 98.28% vs validation 93.04% at epoch 10).
- ResNet50 and MobileNetV2 reached their best validation results at epochs 7 and 9, after which validation performance dropped, so the best checkpoints were restored.
- EfficientNetB0 peaked at epoch 1 and stopped early after epoch 4.
- MobileNetV2 reached similar accuracy to the much larger models with far fewer trainable parameters.

Full per-class metrics, confusion matrices and learning curves are in the report and in the `results/` folder.

---

## 11. Saved Outputs

Each `results/<Model>/` folder contains, where available:

- Trained Keras model / best checkpoint (e.g. `mobilenetv2_best.keras`)
- Training history
- Accuracy and loss curves
- Confusion matrix
- Classification report and per-class metrics
- Test predictions and probabilities
- Overall metrics summary
- Settings/configuration file

---

## 12. Reproducibility

- A fixed random seed of **42** is used for the dataset split, shuffling and model training.
- The same shared CSV split files (`train.csv`, `val.csv`, `test.csv`) are used by all four notebooks.
- The test set is never passed to `model.fit()` and is not used for hyperparameter selection.
- Configuration values (image size, batch size, learning rate, optimizer, loss, epochs, preprocessing) are recorded in each notebook and settings file.
- Results come from a single training run per model. GPU non-determinism may cause small differences on re-runs.

---

## 13. Limitations

- One public dataset only; no external or multi-centre validation.
- Test images come from the same source dataset as the training data.
- Preprocessing, learning rate and augmentation differ across models, and no augmentation ablation was performed.
- Single run per model (no mean/standard deviation over multiple seeds).
- Images were resized to a square without preserving aspect ratio.
- Not a clinical diagnostic system.

## 14. Future Work

- Repeat experiments over multiple seeds and report mean ± standard deviation.
- Run an augmentation ablation study.
- Try progressive unfreezing with smaller learning rates for the transfer-learning models.
- Evaluate on external datasets.
- Analyse misclassified Glioma/Meningioma images.

---

## 15. References and Acknowledgements

- M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle.
- Pretrained ImageNet weights for MobileNetV2, ResNet50 and EfficientNetB0 from `tf.keras.applications`.
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Tan and Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
- Kingma and Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015.
- Srivastava et al., *Dropout*, JMLR 2014.

The full reference list is in the project report.
AI-assisted content was used for drafting and editing support and was reviewed by the group.

---

## 16. Project Purpose

Developed for the **SE4050 – Deep Learning 2026** assignment (supervised deep learning category: four distinct architectures compared under fair, comparable conditions).
