# SE4050 – Brain Tumor Classification

This repository contains the implementation and experimental results for the SE4050 Deep Learning assignment.

The project investigates brain tumor classification from MRI images using four deep learning models:

- Custom CNN
- MobileNetV2
- ResNet50
- EfficientNetB0

The models classify MRI images into four classes:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Dataset

The project uses the Brain Tumor MRI Dataset available from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset is divided into training, validation, and test sets. The test set is kept separate and is used only for final model evaluation.

The dataset itself is not included in this repository. The dataset-access link above should be used to obtain the original images.

## Environment

The experiments were developed using:

- Python
- Google Colab
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Setup

1. Clone this repository.

2. Download the dataset from the Kaggle link provided above.

3. Place the dataset in the required directory or update the dataset paths in the notebooks.

4. Open the required model notebook in Google Colab or a compatible Jupyter environment.

5. Install the required dependencies using:

```bash
pip install -r requirements.txt
