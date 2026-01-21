# Linear Regression Model

## Overview
This file implements a simple linear regression model using PyTorch. Linear regression is a fundamental machine learning algorithm used to model the relationship between input features and continuous target values.

## Model Architecture
The `LinearRegressionModel` class defines a simple linear model:
- **Inputs**: Single feature (scalar)
- **Learnable Parameters**: 
  - `weights`: Slope parameter (w)
  - `bias`: Intercept parameter (b)
- **Output**: Linear prediction: `y = w*x + b`

## Key Functions

### `LinearRegressionModel()`
A custom PyTorch module that implements linear regression with learnable weights and bias.

```python
model = LinearRegressionModel()
```

### `train_model()`
Trains the linear regression model on training data and evaluates on test data.

**Parameters:**
- `model`: PyTorch model to train
- `X_train`, `y_train`: Training data
- `X_test`, `y_test`: Test data
- `loss_fn`: Loss function (typically Mean Absolute Error)
- `optimizer`: Optimization algorithm
- `epochs`: Number of training iterations

**Output:**
- Prints training progress every 10 epochs
- Saves trained model to `models/linear_regression_model.pth`

## Training Details
- **Data**: Generated dummy data using `torch.arange(0, 2, 0.02)`
- **Loss Function**: Mean Absolute Error (MAE)
- **Random Seed**: 42 (for reproducibility)
- **Model Saved**: `models/linear_regression_model.pth`

## Usage Example
```python
# Training setup
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train model
train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=100
)
```

## Output
- Epoch-wise training and test MAE values
- Saved model weights and bias in `models/linear_regression_model.pth`

## Requirements
- PyTorch
- pathlib (standard library)
