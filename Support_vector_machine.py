import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# -----------------------------
# SVM Model
# -----------------------------
class SupportVectorMachineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias  # decision score


# -----------------------------
# Hinge Loss
# -----------------------------
def hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.clamp(1 - y_true * y_pred, min=0))


# -----------------------------
# Training Function
# -----------------------------
def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    optimizer,
    epochs,
    C=1.0
):
    for epoch in range(epochs):

        # ---- Training ----
        model.train()
        scores = model(X_train)

        loss = C * hinge_loss(scores, y_train) + 0.5 * torch.sum(model.weights ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Testing ----
        model.eval()
        with torch.no_grad():
            test_scores = model(X_test)
            test_loss = hinge_loss(test_scores, y_test)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f}"
            )

    # Save model
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH / "svm_model.pth")
    print("\nTraining complete.")


# -----------------------------
# Data
# -----------------------------
torch.manual_seed(42)

x = torch.empty(200, 1).uniform_(-50, 50)
y = torch.where(x > 5, 1.0, -1.0)  # SVM labels {-1, +1}

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

print(f"y_train:{y_train}, y_test: {y_test}")
# -----------------------------
# Train
# -----------------------------
model = SupportVectorMachineModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_model(
    model=model,
    X_train=x_train,
    y_train=y_train,
    X_test=x_test,
    y_test=y_test,
    optimizer=optimizer,
    epochs=100,
    C=1.0
)

# -----------------------------
# Inference
# -----------------------------
model.eval()
with torch.no_grad():
    sample = torch.tensor([4.0, 6.0])
    scores = model(sample)
    predictions = torch.sign(scores)

print("\nPredictions:")
for x_val, s, p in zip(sample, scores, predictions):
    print(f"x={x_val.item():.1f}, score={s.item():.3f}, class={int(p.item())}")
