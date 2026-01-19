import torch
import torch.nn as nn
from pathlib import Path

# -----------------------------
# Dummy Data Creation
# -----------------------------
x = torch.arange(0, 2, 0.02)
y = torch.arange(0, 2, 0.02)

# -----------------------------
# Linear Regression Model
# -----------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Learnable parameters
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


# -----------------------------
# Training Function
# -----------------------------
def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int
):
    """Trains a PyTorch model and evaluates it on test data."""

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):

        # ---- Training ----
        model.train()

        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Testing ----
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())
            print(
                f"Epoch: {epoch} | "
                f"Train MAE: {loss.item():.4f} | "
                f"Test MAE: {test_loss.item():.4f}"
            )

    # -----------------------------
    # Save Model
    # -----------------------------
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "linear_regression_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"\nSaving model to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)

model = LinearRegressionModel()

# -----------------------------
# Train-Test Split (80/20)
# -----------------------------
split_idx = int(0.8 * len(x))

x_train = x[:split_idx].float()
y_train = y[:split_idx].float()
x_test = x[split_idx:].float()
y_test = y[split_idx:].float()

# -----------------------------
# Train the Model
# -----------------------------
train_model(
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    loss_fn=nn.L1Loss(),  # MAE
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    epochs=100
)
