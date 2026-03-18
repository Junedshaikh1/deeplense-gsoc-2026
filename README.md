
---

# Common Test I — Multi-Class Classification

## Task

Build a machine learning model that classifies simulated gravitational lens images into three categories:

- No substructure
- Subhalo substructure
- Vortex substructure

These classes represent different dark matter structure scenarios in strong gravitational lens systems.

---

## Dataset

The dataset consists of simulated gravitational lens images stored as `.npy` arrays.

Each sample is a normalized image representing the light distribution of a lensed galaxy.

Classes included in the dataset:

- **no** → lens with no substructure  
- **sphere** → lens containing subhalo structure  
- **vort** → lens containing vortex-like structure  

---

## Strategy

My approach for this task was:

1. Load the dataset from `.npy` files.
2. Convert the arrays to PyTorch tensors.
3. Add a channel dimension to represent grayscale images.
4. Split the dataset into training and validation sets.
5. Train a convolutional neural network to extract spatial features.
6. Evaluate the model using ROC and AUC metrics.

The goal was to keep the model simple but capable of capturing lens morphology patterns in the images.

---

## Model Architecture

The model is a convolutional neural network composed of:

- Convolution layers for feature extraction
- ReLU activation functions
- Max pooling layers for spatial downsampling
- Fully connected layers for classification

The final layer outputs probabilities for the three lens classes.

---

## Training

The model is trained using:

Loss function  
CrossEntropyLoss

Optimizer  
Adam optimizer

Training involves multiple epochs over the dataset while minimizing classification loss.

---

## Evaluation Metrics

Because class distributions can vary, model performance is evaluated using:

### ROC Curve

The ROC curve measures the trade-off between true positive rate and false positive rate.

### AUC Score

The AUC score summarizes ROC performance into a single number between 0 and 1.

Higher values indicate better classification performance.

### Confusion Matrix

The confusion matrix shows how predictions are distributed across the three classes and helps identify classification errors.

---

# Specific Test V — Lens Finding

## Task

Develop a machine learning model that identifies strong gravitational lenses in observational survey images.

The dataset contains both lens systems and non-lensed galaxies.

The goal is to automatically distinguish lens candidates from normal galaxies.

---

## Dataset

Images are organized into four directories:

train_lenses  
train_nonlenses  
test_lenses  
test_nonlenses

Each image contains three channels corresponding to telescope filters.

Image shape:
