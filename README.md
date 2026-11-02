# Aerial Building Footprint Segmentation

A deep learning project for extracting building footprints from high-resolution aerial and satellite imagery using **PyTorch, U-Net and ResNet34**.

The project uses semantic segmentation to identify building pixels and generate binary building masks. It also explores auxiliary boundary supervision to improve the representation of building edges.

## Project Overview

Building footprint extraction is an important computer vision task used in urban planning, geographical information systems (GIS), infrastructure monitoring and remote sensing.

This project implements a U-Net-based segmentation pipeline trained on the Inria Aerial Image Labeling Dataset. A ResNet34 encoder initialized with ImageNet-pretrained weights is used to extract image features.

The implementation includes:
- A custom PyTorch dataset loader for aerial images and binary masks.
- City-based training and validation splitting.
- Synchronized image and mask augmentation using Albumentations.
- Dual-output U-Net architecture for building-mask and boundary prediction.
- Combined Dice-BCE and boundary supervision loss.
- Mixed-precision model training.
- Overlapping tiled inference for full-resolution aerial images.
- Pixel-level evaluation using IoU, Dice, Precision and Recall.

## Results

<details>
<summary><strong>Validation performance — Vienna (180 full-resolution images)</strong></summary>

| Metric | Result |
|---|---:|
| Intersection over Union (IoU) | **0.6863** |
| Dice Coefficient | **0.8140** |
| Precision | **0.8026** |
| Recall | **0.8256** |

</details>

The results were obtained by evaluating all 180 images in the held-out Vienna city using overlapping tiled inference.

**Important:** Vienna was excluded from training and used for validation and model selection. These are validation results, not results from an independent test set.

## Dataset

**Dataset:** Inria Aerial Image Labeling Dataset

**Source:** [Kaggle — Inria Aerial Image Labeling Dataset](https://www.kaggle.com/datasets/huanranye/inria-aerial-image-labeling-dataset)

The project uses the `cropped_1024_padding` variant containing 1024 × 1024 image patches.

### Dataset organization

Each city folder contains:
- `images/` — RGB aerial images in JPG format.
- `binary_masks/` — Binary building segmentation masks in TIFF format.

### Training and validation split

A city-based split was used to evaluate generalization to geographically unseen imagery.

| Split | Cities | Images |
|---|---|---:|
| Training | Austin, Chicago, Kitsap, Tyrol-w | 720 |
| Validation | Vienna | 180 |
| **Total** | **5 cities** | **900** |

Image and mask files were matched using their filenames.

## Model Architecture

The project uses a U-Net architecture with a ResNet34 encoder.

### Architecture components

- **Encoder:** ResNet34 with ImageNet-pretrained weights.
- **Decoder:** U-Net decoder for spatial feature reconstruction.
- **Input:** Three-channel RGB image.
- **Output:** Two channels:
  - Channel 0: Building segmentation mask.
  - Channel 1: Auxiliary boundary prediction.

### Architecture workflow

```text
       Aerial RGB Image
         (3 × 512 × 512)
                |
                v
       ResNet34 Encoder
       (ImageNet Weights)
                |
                v
          U-Net Decoder
                |
                v
       Dual-Output Prediction
          /           \
         v             v
  Building Mask    Boundary Map
         |             |
         v             v
     Dice + BCE    Boundary BCE
          \           /
           \         /
            v       v
           Combined Loss
```

The auxiliary boundary output provides additional supervision for learning building edges alongside the primary segmentation task.

## Data Preprocessing and Augmentation

The custom dataset implementation handles image and mask loading, preprocessing and augmentation.

### Preprocessing
- RGB image loading using OpenCV.
- TIFF mask loading using OpenCV.
- Conversion of masks into binary values (0 and 1).
- Generation of auxiliary boundary targets using morphological operations.
- Image normalization using ImageNet mean and standard deviation.

### Data augmentation

Albumentations is used for synchronized image and mask transformations.

Training augmentations include:
- Random cropping (512 × 512).
- Horizontal flipping.
- Vertical flipping.
- Random 90-degree rotations.
- Random brightness and contrast adjustments.

Validation preprocessing uses center cropping and normalization without random augmentation.

## Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Model | U-Net |
| Encoder | ResNet34 |
| Input size | 512 × 512 |
| Batch size | 2 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| LR scheduler | ReduceLROnPlateau |
| Mixed precision | Enabled |
| Training hardware | NVIDIA Tesla T4 GPU |

### Loss Function

The training objective combines segmentation loss and auxiliary boundary loss:

```text
Total Loss = Mask Loss + 2.0 × Boundary BCE Loss

Mask Loss = Dice Loss + BCE Loss
```

The boundary loss weight is set to 2.0 to provide additional supervision for boundary prediction.

The best checkpoint was selected using validation loss.

## Full-Resolution Inference

Since the model operates on 512 × 512 patches, larger 1024 × 1024 aerial images are processed using overlapping tiled inference.

### Inference configuration

| Parameter | Value |
|---|---:|
| Patch size | 512 × 512 |
| Stride | 384 pixels |
| Overlap | 128 pixels |
| Aggregation | Average probability maps |

The image is divided into overlapping patches. Each patch is passed through the trained model, and the predicted building probabilities are accumulated and averaged to reconstruct the full-resolution prediction.

This approach reduces boundary discontinuities between neighboring patches.

## Evaluation Metrics

The model is evaluated using pixel-level segmentation metrics.

- **IoU:** Measures the overlap between predicted and ground-truth building regions.
- **Dice:** Measures the similarity between predicted and ground-truth masks.
- **Precision:** Measures the proportion of predicted building pixels that are correct.
- **Recall:** Measures the proportion of actual building pixels successfully detected.

Global pixel counts (TP, FP and FN) are aggregated across validation images before computing the metrics.

## Project Structure

```text
Aerial-Building-Segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── splits.py
│   │
│   └── models/
│       └── model.py
│
├── checkpoints/
│   └── best_model_boundary.pth
│
├── results/
│   ├── metrics.json
│   ├── training_history.json
│   └── training_curves.png
│
└── notebooks/
```

## Installation

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/Aerial-Building-Segmentation.git

cd Aerial-Building-Segmentation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The dataset must be downloaded separately from Kaggle and placed in the expected directory structure.

**Note:** The training and evaluation workflow was developed and executed in Google Colab. The exact training notebook or script should be included in the repository when preparing a fully reproducible setup.

## Future Improvements

Potential directions for further development include:

- Experimenting with stronger encoder architectures.
- Evaluating alternative segmentation loss functions.
- Improving boundary-aware supervision.
- Testing test-time augmentation (TTA).
- Evaluating generalization on additional geographical regions.
- Exploring instance segmentation to distinguish individual buildings.

## Acknowledgements

This project builds upon the starting codebase from:

- [Spacenet Project — Original Reference Repository](https://github.com/TanviDeve-706/Spacenet_project)
- [Inria Aerial Image Labeling Dataset](https://www.kaggle.com/datasets/huanranye/inria-aerial-image-labeling-dataset)
- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)

The dataset preparation, custom data loading and splitting components, training workflow, boundary-supervision implementation and evaluation described above were developed and executed as part of this project.

---

**Project Status:** Initial implementation completed. Further experimentation and improvements are planned.