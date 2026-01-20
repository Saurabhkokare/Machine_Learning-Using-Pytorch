import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Learnable parameters
        self.weights = nn.Parameter(
            torch.randn(1,dtype=torch.float),
            requires_grad=True
        )

        self.bias = nn.Parameter(
            torch.randn(1,dtype=torch.float),
            requires_grad=True
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        linear_combination = self.weights * x + self.bias
        return torch.sigmoid(linear_combination)        ## Apply sigmoid function to get probabilities between 0 and 1 for logistic regression


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
                f"Train Loss: {loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f}"
            )

    # -----------------------------
    # Save Model
    # -----------------------------
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "Logistic_regression_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"\nSaving model to: {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return epoch_count, train_loss_values, test_loss_values

x = torch.arange(0, 2, 0.02)
y = torch.where(x < 1, 0.0, 1.0)  # Binary target: 0 for x<1, 1 for x>=1

## Split data into training and testing sets (80/20 split)
num = int(0.8 * len(x))

x_train,x_test = x[:num],x[num:]
y_train,y_test = y[:num],y[num:]

model = LogisticRegressionModel()
train_model(
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    loss_fn=nn.BCELoss(),  # Binary Cross Entropy Loss for logistic regression
    optimizer=optim.SGD(model.parameters(), lr=0.1),
    epochs=100
)
model.eval()
with torch.inference_mode():
    logits = model(torch.tensor([0.5]))
    probability = torch.sigmoid(logits)

print(f"Predicted probability for x=0.5: {probability.item():.4f}")