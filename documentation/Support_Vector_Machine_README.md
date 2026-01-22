# Support Vector Machine (SVM) Model

## Overview
This file implements a Support Vector Machine for binary classification using PyTorch. SVM is a powerful algorithm that finds the optimal hyperplane to maximize the margin between two classes.

## Model Architecture
The `SupportVectorMachineModel` class implements a linear SVM:
- **Inputs**: Single feature (scalar)
- **Learnable Parameters**: 
  - `weights`: Weight parameter (w)
  - `bias`: Bias parameter (b)
- **Output**: Linear score: `w*x + b`
  - Positive score → class +1
  - Negative score → class -1

## Key Functions

### `SupportVectorMachineModel()`
A custom PyTorch module implementing the SVM decision function.

```python
model = SupportVectorMachineModel()
```

### `hinge_loss()`
Computes the hinge loss, which is the standard loss function for SVM.

```python
loss = hinge_loss(y_pred, y_true)
```

**Formula**: `max(0, 1 - y_true * y_pred)`
- Encourages correct classification with margin of at least 1
- Zero loss if prediction is correct with sufficient margin

### `train_model()`
Trains the SVM model with regularization and evaluates on test data.

**Parameters:**
- `model`: PyTorch SVM model to train
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `optimizer`: Optimization algorithm
- `epochs`: Number of training iterations
- `C`: Regularization parameter (default=1.0)

**Output:**
- Prints training progress every 10 epochs
- Plots training vs test loss curves
- Saves trained model to `models/svm_model.pth`

## Training Details
- **Data**: 200 random samples in range [-50, 50]
  - Class +1: if x > 5
  - Class -1: if x ≤ 5
- **Loss Function**: Hinge Loss + L2 Regularization
  - `Total Loss = C * hinge_loss + 0.5 * ||weights||²`
- **Random Seed**: 42 (for reproducibility)
- **Model Saved**: `models/svm_model.pth`

## Usage Example
```python
# Data preparation
torch.manual_seed(42)
x = torch.empty(200, 1).uniform_(-50, 50)
y = torch.where(x > 5, 1.0, -1.0)

# Initialize model
model = SupportVectorMachineModel()

# Train model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_model(
    model=model,
    X_train=x,
    y_train=y,
    X_test=x_test,
    y_test=y_test,
    optimizer=optimizer,
    epochs=100,
    C=1.0
)
```

## Output
- Loss curves plotted (training vs test loss across epochs)
- Saved model weights and bias in `models/svm_model.pth`
- Console output showing training progress

## Key Concepts
- **Hinge Loss**: Penalizes misclassifications and points within the margin
- **Margin**: Distance between decision boundary and closest data points
- **Regularization (C parameter)**: 
  - Higher C: stricter margin constraint (may overfit)
  - Lower C: more tolerance for margin violations (may underfit)
- **Binary Classification**: Uses +1 and -1 labels

## Visualization
The model plots training and test loss over epochs to visualize:
- Model convergence
- Overfitting/underfitting behavior
- Training stability

## Requirements
- PyTorch
- matplotlib
- pathlib (standard library)
