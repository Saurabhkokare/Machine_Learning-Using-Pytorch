# Logistic Regression Model

## Overview
This file implements logistic regression using PyTorch. Logistic regression is used for binary classification tasks, outputting probabilities between 0 and 1.

## Model Architecture
The `LogisticRegressionModel` class defines a logistic regression model:
- **Inputs**: Single feature (scalar)
- **Learnable Parameters**: 
  - `weights`: Slope parameter (w)
  - `bias`: Intercept parameter (b)
- **Output**: Sigmoid activation applied to linear combination: `σ(w*x + b)`
  - Outputs probability between 0 and 1 for binary classification

## Key Functions

### `LogisticRegressionModel()`
A custom PyTorch module implementing logistic regression with sigmoid activation.

```python
model = LogisticRegressionModel()
```

### `train_model()`
Trains the logistic regression model on training data and evaluates on test data.

**Parameters:**
- `model`: PyTorch model to train
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `loss_fn`: Loss function (typically Binary Cross Entropy)
- `optimizer`: Optimization algorithm
- `epochs`: Number of training iterations

**Output:**
- Prints training progress every 10 epochs
- Saves trained model to `models/Logistic_regression_model.pth`
- Returns epoch counts, training losses, and test losses

## Training Details
- **Data**: Binary classification where class 0 if x<1, class 1 if x≥1
- **Data Split**: 80/20 train-test split
- **Loss Function**: Binary Cross Entropy Loss (BCELoss)
- **Model Saved**: `models/Logistic_regression_model.pth`

## Usage Example
```python
# Data preparation
x = torch.arange(0, 2, 0.02)
y = torch.where(x < 1, 0.0, 1.0)  # Binary target

num = int(0.8 * len(x))
x_train, x_test = x[:num], x[num:]
y_train, y_test = y[:num], y[num:]

# Initialize model
model = LogisticRegressionModel()

# Train model
epoch_count, train_losses, test_losses = train_model(
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    loss_fn=nn.BCELoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    epochs=100
)
```

## Output
- Epoch-wise training and test loss values
- Saved model weights and bias in `models/Logistic_regression_model.pth`
- Returns lists of epoch counts and loss values for plotting/analysis

## Key Concepts
- **Sigmoid Function**: Maps any real number to value between 0 and 1
- **Binary Cross Entropy**: Appropriate loss function for binary classification
- **Probability Output**: Model outputs can be interpreted as classification probability

## Requirements
- PyTorch
- pathlib (standard library)
